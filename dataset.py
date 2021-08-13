import os
import glob

import torch
from torch.utils.data import Dataset
from scipy import signal
from scipy.io import wavfile
import cv2
from PIL import Image
import numpy as np
"""
./data
    img/
        001.jpg
        ...
    img/test/
        002.jpg
        ...
    valid.txt
    train.txt
    test.txt
"""

class TextDataset(Dataset):
    # word to label and label to word
    CHARS = []
    with open('word.txt','r') as fr:
        Lines = fr.readlines()
        for w in Lines:
            CHARS.append(w[:-1])
    CHAR2LABEL = {char: i + 1 for i, char in enumerate(CHARS)}
    LABEL2CHAR = {label: char for char, label in CHAR2LABEL.items()}

    def __init__(self, root_dir,txt_path, img_height, img_width):
        # root_dir = ./data/ ,txt_path = 'train.txt'
        self.imgs, self.labels = self._load_from_raw_files(root_dir,txt_path)
        self.img_height = img_height
        self.img_width = img_width

    def _load_from_raw_files(self, root_dir,txt_path):

        imgs,labels = [],[]
        with open(os.path.join(root_dir,txt_path), 'r') as fr:
            for line in fr.readlines():
                # print(line)
                line = line.strip().split(',')
                path, label = line[0] , line[1]
                img = os.path.join(root_dir,'img', path)
                imgs.append(img)
                labels.append(label)
        return imgs, labels

    def __len__(self):
        return len(self.imgs)

    def __getitem__(self, index):

        path = self.imgs[index]
        image = Image.open(path).convert('L')  # gray-scale
        image = image.resize((self.img_width, self.img_height), resample=Image.BILINEAR)
        image = np.array(image)
        image = image.reshape((1, self.img_height, self.img_width))
        image = (image / 127.5) - 1.0
        image = torch.FloatTensor(image)

        label = self.labels[index]
        target = [self.CHAR2LABEL[c] for c in label]
        target_length = [len(target)]

        target = torch.LongTensor(target)
        target_length = torch.LongTensor(target_length)
        return image, target, target_length


def text_collate_fn(batch):
    images, targets, target_lengths = zip(*batch)
    images = torch.stack(images, 0)
    targets = torch.cat(targets, 0)
    target_lengths = torch.cat(target_lengths, 0)
    return images, targets, target_lengths


class PredictDataset(Dataset):
    CHARS = []
    with open('word.txt','r') as fr:
        Lines = fr.readlines()
        for w in Lines:
            CHARS.append(w[:-1])
    CHAR2LABEL = {char: i + 1 for i, char in enumerate(CHARS)}
    LABEL2CHAR = {label: char for char, label in CHAR2LABEL.items()}

    def __init__(self, root_dir,txt_path, img_height, img_width):
        # root_dir = ./data/ ,txt_path = 'test.txt'
        self.imgs = self.load_data(root_dir,txt_path)
        self.img_height = img_height
        self.img_width = img_width

    def load_data(self, root_dir,txt_path):
        
        imgs = []
        with open(os.path.join(root_dir,txt_path), 'r') as fr:
            for line in fr.readlines():
                path = line.strip().split(',')[0]
                img = os.path.join(root_dir,'img_test', path)
                imgs.append(img)
        return imgs

    def __len__(self):
        return len(self.imgs)

    def __getitem__(self, index):
        
        path = self.imgs[index]       
        image = Image.open(path).convert('L')  
        image = image.resize((self.img_width, self.img_height), resample=Image.BILINEAR)
        image = np.array(image)
        image = image.reshape((1, self.img_height, self.img_width))
        image = (image / 127.5) - 1.0
        image = torch.FloatTensor(image)

        return image