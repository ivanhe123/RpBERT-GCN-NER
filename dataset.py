import torch
from torch.utils.data import Dataset
from typing import List, Optional
from PIL import Image
from torchvision import transforms
from tqdm import tqdm
import torch_geometric.data as g_dat
import spacy


class DataPoint:
    def __init__(self):
        self.feat: Optional[torch.Tensor] = None
        self.label: Optional[int] = None


class Token(DataPoint):
    def __init__(self, text, label):
        super().__init__()
        self.text: str = text
        self.label = label


class Sentence(DataPoint):
    def __init__(self, tokens: List[Token] = None, text: str = None):
        super().__init__()
        self.tokens: List[Token] = tokens
        self.text = text

    def __len__(self):
        return len(self.tokens)

    def __getitem__(self, index: int):
        return self.tokens[index]

    def __iter__(self):
        return iter(self.tokens)

    def __str__(self):
        return self.text if self.text else ' '.join([token.text for token in self.tokens])


class ImageData(DataPoint):
    def __init__(self, file_name: str):
        super().__init__()
        #print(file_name)
        self.file_name: str = file_name
        self.data: ImageData = None


class Data(DataPoint):
    def __init__(self, sentence, image, label=-1):
        super().__init__()
        self.sentence: Sentence = sentence
        self.image: ImageData = image
        self.label = label




class CustomDataset(Dataset):
    def __init__(self, datas: List[Data], path_to_images: str, load_image: bool = True):
        self.datas: List[Data] = datas
        self.path_to_images = path_to_images
        self.load_image = load_image
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])

    def __len__(self):
        return len(self.datas)

    def __getitem__(self, index: int):

        data = self.datas[index]

        if self.load_image:
            image = data.image

            if image.data is not None or image.feat is not None:
                return data
            # print(image.file_name)
            path_to_image = self.path_to_images + "/" + image.file_name
            image.data = Image.open(path_to_image).convert('RGB')
            image.data = self.transform(image.data)
        return data


class Corpus:
    def __init__(self, train=None, dev=None, test=None):
        self.train: CustomDataset = train
        self.dev: CustomDataset = dev
        self.test: CustomDataset = test
