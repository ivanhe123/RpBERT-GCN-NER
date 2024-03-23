import re
import csv
from pathlib import Path
from dataset import Token, Sentence, ImageData, Data, CustomDataset, Corpus
import constants as constants

# constants for preprocessing
SPECIAL_TOKENS = ['\ufe0f', '\u200d', '\u200b', '\x92']
IMGID_PREFIX = 'IMGID:'
URL_PREFIX = 'http://t.co/'
UNKNOWN_TOKEN = '[UNK]'


def normalize_text(text: str):
    url_re = r' http[s]?://t.co/\w+$'
    text = re.sub(url_re, '', text)
    return text


def load_itr_corpus(path: str, split: int = 3576, normalize: bool = True):
    path = Path(path)
    path_to_images = path / 'images'
    assert path.exists()
    assert path_to_images.exists()

    with open(path/'data.csv', encoding='utf-8') as csv_file:
        csv_reader = csv.DictReader(csv_file, doublequote=False, escapechar='\\')
        datas = [Data(
            sentence=Sentence(text=normalize_text(row['tweet']) if normalize else row['tweet']),
            image=ImageData(f"T{row['tweet_id']}.jpg"),
            label=int(row['image_adds'])
        ) for row in csv_reader]

    train = CustomDataset(datas[:split], path_to_images)
    test = CustomDataset(datas[split:], path_to_images)
    return Corpus(train=train, test=test)


def load_ner_dataset(path_to_txt, path_to_images, load_image: bool = True) -> CustomDataset:
    tokens = []
    image_id = None
    datas = []

    with open(str(path_to_txt), encoding='utf-8') as txt_file:
        for line in txt_file:
            line = line.rstrip()

            if line.startswith(IMGID_PREFIX):
                image_id = line[len(IMGID_PREFIX):]
            elif line != '':
                text, label = line.split('\t')
                if text == '' or text.isspace() or text in SPECIAL_TOKENS or text.startswith(URL_PREFIX):
                    text = UNKNOWN_TOKEN
                tokens.append(Token(text, constants.LABEL_TO_ID[label]))
            else:
                datas.append(Data(Sentence(tokens), ImageData(f'{image_id}.jpg')))
                tokens = []
    datas.append(Data(Sentence(tokens), ImageData(f'{image_id}.jpg')))
    return CustomDataset(datas, path_to_images, load_image)


def load_ner_corpus(path: str, load_image: bool = True) -> Corpus:
    path_to_train_file = path + '/train.txt'
    path_to_dev_file = path + '/dev.txt'
    path_to_test_file = path + '/test.txt'
    path_to_images = path + '/images'

    train = load_ner_dataset(path_to_train_file, path_to_images, load_image)
    dev = load_ner_dataset(path_to_dev_file, path_to_images, load_image)
    test = load_ner_dataset(path_to_test_file, path_to_images, load_image)

    return Corpus(train, dev, test)
