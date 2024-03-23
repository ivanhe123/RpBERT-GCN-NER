# GCN-RpBert
This is an implementation of the RpBERT paper: [RpBERT: A Text-image Relation Propagation-based BERT Model for Multimodal NER](https://arxiv.org/abs/2102.02967), With a additional layer of GCN added in between the bert-resnet encode layer and the Bi-LSTM layer.

Original Code (without the GCN layer) from [MultiModel-NER/RpBERT Repo](https://github.com/Multimodal-NER/RpBERT).

The dataset has already been installed.
Also torchcrf is already in the directory, since for some reason, PIP kept downloading the crf package instead of the torch-crf package even thought I entered pip install torch-crf.


## Warning
This repository is still under development, if there are any questions, feel free to just start and issue.

The cuda support is turned off. But if you want to turn it on, it is in the file model.py. Find this code snippet in the class MultimodelModel and switch the code:

```python
...
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
...
```

## Instructions
### Installing
Install the packages in requirements.txt
### Downloading Resnet152
run the following code:
```python
import torchvision
resnet152_model = torchvision.models.resnet152(from_pretrained=True)
```
Copy the resnet152.pth model under the directory resources/models/cnn after it finished downloading. It should be in the path C:/Users/Administrator/.resnet152/weights for windows.

### Downloading Bert-Base-Uncased
Dowload via link: [Bert-Base-Uncased](https://storage.googleapis.com/bert_models/2020_02_20/uncased_L-12_H-768_A-12.zip).
Unzip the package into a dir, and move it under the directory resources/models/transformers.
After moving it to the directory resources/models/transformers, download the pytorch_model.bin and config.json manually into the directory from the [bert-base-uncased hugginface page](https://huggingface.co/bert-base-uncased/tree/main).

### Downloading the dataset
The twitter2015 paper: [(Zhang et al., 2018)](http://qizhang.info/paper/aaai2017-twitterner.pdf).

The [dataset](http://qizhang.info/paper/data/aaai2018_multimodal_NER_data.zip) need to downloaded and unziped. Then change the name into "twitter2015". Then go into the directory and change the directory containing all the images to "images". After all this, move the directory under the path: resources/dataset.
### Training
To train, run train.py. An example on how to run the model is at the file main.py
