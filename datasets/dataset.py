import re
import json
import math
import torch
from pathlib import Path
from torch.utils.data import Dataset
from PIL import Image
import torchvision.transforms as transforms

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
                self.img_labels.append((self.img_dir / img_file, token, label))

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
            img_labels.append((self.img_dir / img_file, token, label))

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
        img_path, label, text = self.img_labels[idx]
        image = Image.open(img_path).convert('RGB')
        
        if self.transform:
            image = self.transform(image)
        
        input_token = [self.vocab['token2id']['SOS']] + label[:self.vocab['max_length']]
        label = label[:self.vocab['max_length']] + [self.vocab['token2id']['EOS']]

        input_token = torch.as_tensor(self.pad_list(input_token, self.vocab['max_length'], self.vocab['token2id']['PAD']))
        label = torch.as_tensor(self.pad_list(label, self.vocab['max_length'], self.vocab['token2id']['PAD']))

        return image, input_token, label, torch.as_tensor(self.lang_dict['token2id'][self.language]), text
    

class Image_Pad(object):
    def __init__(self, max_size, PAD_type='right'):
        self.toTensor = transforms.ToTensor()
        self.max_size = max_size
        self.max_width_half = math.floor(max_size[2] / 2)
        self.PAD_type = PAD_type

    def __call__(self, img):
        img = self.toTensor(img)
        c, h, w = img.size()
        Pad_img = torch.FloatTensor(*self.max_size).fill_(0)
        Pad_img[:, :, :w] = img  # right pad
        if self.max_size[2] != w:  # add border Pad
            Pad_img[:, :, w:] = img[:, :, w - 1].unsqueeze(2).expand(c, h, self.max_size[2] - w)

        return Pad_img

class Custom_Collate(object):
    def __init__(self, imgH=112, imgW=448):
        self.imgH = imgH
        self.imgW = imgW

        self.normalize = transforms.Compose([
            transforms.Grayscale(),
            transforms.Normalize(mean=[0.5,], std=[0.5,])
        ])

    def __call__(self, batch):
        batch = filter(lambda x: x is not None, batch)
        images, input_tokens, labels, lang_ids, text = zip(*batch)

        resized_max_w = self.imgW
        transform = Image_Pad((3, self.imgH, resized_max_w))

        resized_images = []
        for idx, image in enumerate(images):
            w, h = image.size
            ratio = w / float(h)
            if math.ceil(self.imgH * ratio) > self.imgW:
                resized_w = self.imgW
            else:
                resized_w = math.ceil(self.imgH * ratio)

            resized_image = image.resize((resized_w, self.imgH), Image.BICUBIC)

            transformed_image = transform(resized_image)
            normalized_image = self.normalize(transformed_image)
            resized_images.append(normalized_image)

        image_tensors = torch.cat([t.unsqueeze(0) for t in resized_images], 0)

        return image_tensors, torch.stack(input_tokens), torch.stack(labels), torch.stack(lang_ids), [text]