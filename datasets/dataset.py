import re
import json
import torch
from pathlib import Path
from torch.utils.data import Dataset
from PIL import Image

class CustomImageDataset(Dataset):
    def __init__(self, base_dir, img_dir, data_list, transform=None, dataset_type='train', regex=None, lang_dict=None):
        self.img_labels = []
        self.base_dir = Path(base_dir)
        self.data_list = data_list
        self.img_dir = self.base_dir / img_dir
        self.regex = regex
        self.language = img_dir

        self.dataset_type = dataset_type
        self.transform = transform

        self.vocab_file = self.base_dir / (img_dir + '.json')
        self.vocab = {'id2token' : {0:"PAD",
                                    1:"SOS",
                                    2:"EOS",
                                    3:"OOV"},
                      'token2id' : {"PAD":0,
                                    "SOS":1,
                                    "EOS":2,
                                    "OOV":3},
                      'max_length' : 63}

        self.lang_dict = lang_dict

        if self.vocab_file.exists():
            with self.vocab_file.open('r') as f:
                self.vocab = json.load(f)

        # 레이블 파일들을 읽고 이미지-레이블 쌍 생성
        if self.dataset_type == 'train':
            self.img_labels = self.update_vocab()
        else:
            for line in self.data_list:
                img_file, origin_label = line.strip().split('\t')
                token = []
                if self.regex is not None:
                    label = re.sub(self.regex, '', origin_label).lower()
                    if len(label) != len(origin_label):
                        continue
                else:
                    label = origin_label.lower()

                for char in list(label):
                    if char not in self.vocab['token2id']:
                        token.append(self.vocab['token2id']['OOV']) # add OOV
                    else:
                        token.append(self.vocab['token2id'][char])
                self.img_labels.append((self.img_dir / img_file, token))

    def update_vocab(self):
        img_labels = []
        for line in self.data_list:
            img_file, origin_label = line.strip().split('\t')
            token = []
            if self.regex is not None:
                label = re.sub(self.regex, '', origin_label).lower()
                if len(label) != len(origin_label):
                    continue
            else:
                label = origin_label.lower()

            if len(label) > self.vocab['max_length']: # with EOS
                self.vocab['max_length'] = len(label)
            for char in list(label):
                if char not in self.vocab['token2id']:
                    self.vocab['token2id'][char] = len(self.vocab['token2id'])
                    self.vocab['id2token'][len(self.vocab['id2token'])] = char
                token.append(self.vocab['token2id'][char])
            img_labels.append((self.img_dir / img_file, token))

        with self.vocab_file.open('w') as f:
            json.dump(self.vocab, f)

        return img_labels

    def pad_list(self, input_list, max_length, padding_value):
        return input_list + [padding_value] * (max_length - len(input_list) + 1)

    def get_vocab(self):
        return self.vocab

    def __len__(self):
        return len(self.img_labels)

    def __getitem__(self, idx):
        img_path, label = self.img_labels[idx]
        image = Image.open(img_path).convert('RGB')
        
        if self.transform:
            image = self.transform(image)
        
        input_token = [self.vocab['token2id']['SOS']] + label[:self.vocab['max_length']]
        label = label[:self.vocab['max_length']] + [self.vocab['token2id']['EOS']]

        input_token = torch.as_tensor(self.pad_list(input_token, self.vocab['max_length'], self.vocab['token2id']['PAD']))
        label = torch.as_tensor(self.pad_list(label, self.vocab['max_length'], self.vocab['token2id']['PAD']))

        return image, input_token, label, torch.as_tensor(self.lang_dict['token2id'][self.language])