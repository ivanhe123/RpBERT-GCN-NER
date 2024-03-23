from typing import List
import torch
from torch import nn
import torch.nn.functional as F
import torchvision
from transformers import PreTrainedTokenizer, PreTrainedModel, AutoTokenizer, AutoModel, AutoConfig
from dataset import DataPoint, Data
import constants as constants
import spacy
import re
import spacy.displacy as displacy
import string
from torchcrf import CRF
from torch_geometric.nn import GCNConv

# constants for model
CLS_POS = 0
SUBTOKEN_PREFIX = '##'
IMAGE_SIZE = 224
VISUAL_LENGTH = (IMAGE_SIZE // 32) ** 2


def use_cache(module: nn.Module, data_points: List[DataPoint]):
    for parameter in module.parameters():
        if parameter.requires_grad:
            return False
    for data_point in data_points:
        if data_point.feat is None:
            return False
    return True


def resnet_encode(model, x):
    x = model.conv1(x)
    x = model.bn1(x)
    x = model.relu(x)
    x = model.maxpool(x)

    x = model.layer1(x)
    x = model.layer2(x)
    x = model.layer3(x)
    x = model.layer4(x)

    x = x.view(x.size()[0], x.size()[1], -1)
    x = x.transpose(1, 2)

    return x
def clean(sentence):
    sentence = sentence.replace("[UNK]", "")
    url_re = r' http[s]?://t.co/\w+$'
    sentence = re.sub(url_re, '', sentence)
    punctuation = ['.', ',', ';', ':', "(", ")", "'", '"'," "]
    for p in punctuation:
        sentence = sentence.replace(' ' + p, p)
    sentence=sentence.translate(str.maketrans("","",string.punctuation))
    return sentence
def get_edge_index(sentence):
    nlp = spacy.load("en_core_web_sm")
    sentence=clean(sentence)
    doc = nlp(sentence)
    # Create a list of words (nodes) in the sentence
    words = [token.text for token in doc]

    # Create a list of edges based on the dependency relations
    edges = [(token.head.text, token.text) for token in doc if token.text != token.head.text]

    # Convert the edges to indices
    edge_index = [[words.index(src), words.index(dst)] for src, dst in edges]

    # Convert the edge_index list to a PyTorch tensor
    edge_index = torch.tensor(edge_index).t().contiguous()

    return edge_index
class MultiModelModel(nn.Module):
    def __init__(
            self,
            device: torch.device,
            tokenizer: PreTrainedTokenizer,
            encoder_t: PreTrainedModel,
            hid_dim_t: int,
            encoder_v: nn.Module = None,
            hid_dim_v: int = None,
    ):
        super().__init__()
        self.device = device
        self.tokenizer = tokenizer
        self.encoder_t = encoder_t
        self.hid_dim_t = hid_dim_t
        self.encoder_v = encoder_v
        self.hid_dim_v = hid_dim_v
        self.token_embedding = None
        self.proj = nn.Linear(hid_dim_v, hid_dim_t)
        self.aux_head = nn.Linear(hid_dim_t, 2)
        hid_dim_rnn = 256
        num_layers = 2
        num_directions = 2
        self.gcn_layer = GCNConv(self.hid_dim_t, self.hid_dim_t)
        self.rnn = nn.LSTM(self.hid_dim_t, hid_dim_rnn, num_layers, batch_first=True, bidirectional=True)
        self.head = nn.Linear(hid_dim_rnn * num_directions, constants.LABEL_SET_SIZE)
        self.crf = CRF(constants.LABEL_SET_SIZE, batch_first=True)
        self.to(device)

    @classmethod
    def from_pretrained(cls, cuda, t_encoder, v_encoder):
        device = torch.device(f'cpu')
        #device = torch.device(f"cuda:{cuda}")
        models_path = '../resources/models'

        encoder_t_path = f'{models_path}/transformers/{t_encoder}'
        tokenizer = AutoTokenizer.from_pretrained(encoder_t_path)
        encoder_t = AutoModel.from_pretrained(encoder_t_path)
        config = AutoConfig.from_pretrained(encoder_t_path)
        hid_dim_t = config.hidden_size


        encoder_v = getattr(torchvision.models, v_encoder)()
        encoder_v.load_state_dict(torch.load(f'{models_path}/cnn/{v_encoder}.pth'))
        hid_dim_v = encoder_v.fc.in_features




        return cls(
            device=device,
            tokenizer=tokenizer,
            encoder_t=encoder_t,
            hid_dim_t=hid_dim_t,
            encoder_v=encoder_v,
            hid_dim_v=hid_dim_v,
        )

    def _bert_forward_with_image(self, inputs, datas, gate_signal=None):
        images = [data.image for data in datas]
        textual_embeds = self.encoder_t.embeddings.word_embeddings(inputs.input_ids)
        visual_embeds = torch.stack([image.data for image in images]).to(self.device)
        if not use_cache(self.encoder_v, images):
            visual_embeds = resnet_encode(self.encoder_v, visual_embeds)
        visual_embeds = self.proj(visual_embeds)
        if gate_signal is not None:
            visual_embeds *= gate_signal
        inputs_embeds = torch.concat((textual_embeds, visual_embeds), dim=1)

        batch_size = visual_embeds.size()[0]
        visual_length = visual_embeds.size()[1]

        attention_mask = inputs.attention_mask
        visual_mask = torch.ones((batch_size, visual_length), dtype=attention_mask.dtype, device=self.device)
        attention_mask = torch.cat((attention_mask, visual_mask), dim=1)

        token_type_ids = inputs.token_type_ids
        visual_type_ids = torch.ones((batch_size, visual_length), dtype=token_type_ids.dtype, device=self.device)
        token_type_ids = torch.cat((token_type_ids, visual_type_ids), dim=1)

        return self.encoder_t(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            return_dict=True
        )

    def ner_encode(self, datas: List[Data], gate_signal=None):
        sentence_batch = [data.sentence for data in datas]
        tokens_batch = [[token.text for token in sentence] for sentence in sentence_batch]
        inputs = self.tokenizer(tokens_batch, is_split_into_words=True, padding=True, return_tensors='pt',
                                return_special_tokens_mask=True, return_offsets_mapping=True).to(self.device)

        outputs = self._bert_forward_with_image(inputs, datas, gate_signal)
        feat_batch = outputs.last_hidden_state[:, :-VISUAL_LENGTH]


        ids_batch = inputs.input_ids
        offset_batch = inputs.offset_mapping
        mask_batch = inputs.special_tokens_mask.bool().bitwise_not()
        for sentence, ids, offset, mask, feat in zip(sentence_batch, ids_batch, offset_batch, mask_batch, feat_batch):
            ids = ids[mask]
            offset = offset[mask]
            feat = feat[mask]
            subtokens = self.tokenizer.convert_ids_to_tokens(ids)
            length = len(subtokens)

            token_list = []
            feat_list = []
            i = 0
            while i < length:
                j = i + 1

                while j < length and (offset[j][0] != 0 or subtokens[j].startswith(SUBTOKEN_PREFIX)):
                    j += 1
                token_list.append(''.join(subtokens[i:j]))
                feat_list.append(torch.mean(feat[i:j], dim=0))
                i = j
            assert len(sentence) == len(token_list)
            for token, token_feat in zip(sentence, feat_list):
                token.feat = token_feat


    def ner_forward(self, datas: List[Data]):
        tokens_batch = [[token.text for token in data.sentence] for data in datas]
        inputs = self.tokenizer(tokens_batch, is_split_into_words=True, padding=True, return_tensors='pt')
        inputs = inputs.to(self.device)
        outputs = self._bert_forward_with_image(inputs,  datas)
        feats = outputs.last_hidden_state[:, CLS_POS]
        logits = self.aux_head(feats)
        gate_signal = F.softmax(logits, dim=1)[:, 1].view(len(datas), 1, 1)
        self.ner_encode(datas, gate_signal)

        sentences = [data.sentence for data in datas]
        batch_size = len(sentences)
        edges = [get_edge_index(str(sentence)).to(self.device) for sentence in sentences]
        lengths = [len(sentence) for sentence in sentences]
        max_length = max(lengths)

        feat_list = []
        zero_tensor = torch.zeros(max_length * self.hid_dim_t, device=self.device)
        for sentence in sentences:
            feat_list += [token.feat for token in sentence]
            num_padding = max_length - len(sentence)
            if num_padding > 0:
                padding = zero_tensor[:self.hid_dim_t * num_padding]
                feat_list.append(padding)

        feats = torch.cat(feat_list).view(batch_size, max_length, self.hid_dim_t)
        feats_out = []

        for edge, feat,lenth in zip(edges, feats,lengths):
            if lenth > 1:
                feats_out.append(self.gcn_layer(feat, edge))
        feats = torch.stack(feats_out)
        feats = nn.utils.rnn.pack_padded_sequence(feats, lengths, batch_first=True, enforce_sorted=False)
        feats, _ = self.rnn(feats)
        feats, _ = nn.utils.rnn.pad_packed_sequence(feats, batch_first=True)

        logits_batch = self.head(feats)

        labels_batch = torch.zeros(batch_size, max_length, dtype=torch.long, device=self.device)
        for i, sentence in enumerate(sentences):
            labels = torch.tensor([token.label for token in sentence], dtype=torch.long, device=self.device)
            labels_batch[i, :lengths[i]] = labels

        mask = torch.zeros(batch_size, max_length, dtype=torch.bool, device=self.device)
        for i in range(batch_size):
            mask[i, :lengths[i]] = 1
        loss = -self.crf(logits_batch, labels_batch, mask)
        pred_ids = self.crf.decode(logits_batch, mask)
        pred = [[constants.ID_TO_LABEL[i] for i in ids] for ids in pred_ids]

        return loss, pred
