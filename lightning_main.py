import yaml
import torch
import random
import datetime
import lightning as L
import torchvision.transforms as transforms


from pathlib import Path
from models.lightning_model import OcrModel
from torch.utils.data import DataLoader
from datasets.dataset import CustomImageDataset

torch.manual_seed(0)
torch.cuda.manual_seed(0)
random.seed(0)
torch.set_float32_matmul_precision('medium')

train_transforms = transforms.Compose(
    [
        transforms.AutoAugment(transforms.AutoAugmentPolicy.SVHN),
        transforms.Resize((112,224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ]
)

valid_transforms = transforms.Compose(
    [
        transforms.Resize((112,224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ]
)

regex = {
    'C' : r'[^\u4e00-\u9fff]',
    'J' : r'[^\u3040-\u309f\u30a0-\u30ff\u4e00-\u9fff]',
    'K' : r'[^\uac00-\ud7a3]',
    'E' : r'[^a-zA-Z]',
    'M' : None
}

lang_dict = {
        'token2id' : {
            'K' : 1,
            'C' : 2,
            'J' : 3,
            'E' : 4,
            'M' : 5
        },
        'id2token' : {
            "1" : 'K',
            "2" : 'C',
            "3" : 'J',
            "4" : 'E',
            "5" : 'M'
        }
}

device = torch.device("cuda" if torch.cuda.is_available() else 'cpu')

def load_setting(setting):
    with open(setting, 'r', encoding='utf8') as f:
        cfg = yaml.load(f, Loader=yaml.FullLoader)

    return cfg

def save_setting(cfg, save_dir):
    with open(save_dir, 'w', encoding='utf8') as f:
        yaml.dump(cfg, f)

def load_dataset(base_dir):
    dataset_dict = {'train': None, 'valid': None, 'vocab': {}}
    label_list = Path(base_dir).glob('*.txt')

    for label_file in label_list:
        with label_file.open('r') as f:
            data_list = f.readlines()
        total_len = len(data_list)
        train_size = int(total_len * 0.8)
        valid_size = int(total_len * 0.1)

        train_data = data_list[:train_size]
        valid_data = data_list[train_size:train_size + valid_size]
        test_data = data_list[train_size + valid_size:]

        train_dataset = CustomImageDataset(base_dir=base_dir, img_dir=label_file.stem, data_list = train_data, transform=train_transforms, dataset_type='train', regex=regex[label_file.stem], lang_dict=lang_dict)
        valid_dataset = CustomImageDataset(base_dir=base_dir, img_dir=label_file.stem, data_list = valid_data, transform=valid_transforms, dataset_type='valid', regex=regex[label_file.stem], lang_dict=lang_dict)

        dataset_dict['vocab'][label_file.stem] = train_dataset.get_vocab()
        if dataset_dict['train'] is None and dataset_dict['valid'] is None:
            dataset_dict['train'] = train_dataset
            dataset_dict['valid'] = valid_dataset
        else:
            dataset_dict['train'] += train_dataset
            dataset_dict['valid'] += valid_dataset

    return dataset_dict

if __name__ == '__main__':
    cfg = load_setting('./settings/default.yaml')

    dataset_dict = load_dataset(cfg['DATA']['DIR'])

    cfg['vocab'] = dataset_dict['vocab']
    cfg['lang_dict'] = lang_dict

    formatted_time = datetime.datetime.now().strftime('%Y%m%d%H%M')
    save_setting(cfg, f'./settings/default_{formatted_time}.yaml')

    ocr_model = OcrModel(cfg)

    checkpoint_callback = L.pytorch.callbacks.ModelCheckpoint(
        monitor='val_loss', 
        mode='min', 
        save_top_k=3, 
        save_last=True, 
        filename='{epoch}-{val_loss:.2f}',
        save_weights_only=True,
        )
    
    trainer = L.Trainer(accelerator='gpu', devices=torch.cuda.device_count(), 
                         max_epochs=cfg['TRAIN_PARAMS']['EPOCHS'],
                         num_sanity_val_steps=0,
                         strategy='ddp',
                         precision='16-mixed', benchmark=True, gradient_clip_algorithm='norm',
                         callbacks=[checkpoint_callback])

    train_dataloader = DataLoader(dataset_dict['train'], batch_size=cfg['TRAIN_PARAMS']['BATCH_SIZE'], 
                                  num_workers=8, shuffle=True)

    valid_datalodaer = DataLoader(dataset_dict['valid'], batch_size=cfg['TRAIN_PARAMS']['BATCH_SIZE'], 
                                  num_workers=8, shuffle=False)

    trainer.fit(ocr_model, train_dataloaders=train_dataloader, 
                val_dataloaders=valid_datalodaer)