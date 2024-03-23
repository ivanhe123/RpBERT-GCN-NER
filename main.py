from dataset import *
import pickle
encoder_t = "bert-base-uncased"
encoder_v = "resnet152"
model = pickle.load(open(f'trained/{encoder_t}-BiLSTM-{encoder_v}.pkl', "rb"))
input_text = "RT @ThePatriot142: Timeline shows how Clintons took $1.8 Millrom from keystone pipeline investores http://t.co/17CGxAS18d #ClintonCash"
image="62654.jpg"
tokens = [Token(x,-1) for x in input_text.split(" ")]
datas = [Data(Sentence(tokens=tokens), ImageData(image))]
input_data = CustomDataset(datas, "../resources/datasets/twitter2015/images")
_, pred = model.ner_forward(input_data)
print(pred)