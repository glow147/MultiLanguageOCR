import yaml
from models.lightning_model import OcrModel
import torch, random
from torch.utils.data import DataLoader
from datasets.dataset import CustomImageDataset, Custom_Collate
from pathlib import Path
import torchvision.transforms as transforms

torch.manual_seed(4)
torch.cuda.manual_seed(4)
random.seed(4)
torch.set_float32_matmul_precision('medium')

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

def load_setting(setting):

    with open(setting, 'r', encoding='utf8') as f:
        cfg = yaml.load(f, Loader=yaml.FullLoader)

    return cfg

cfg = load_setting(['trained.yaml'])
pth_file = ''
base_dir = '' # same as training

device = torch.device('cuda:0')

ocr_model = OcrModel.load_from_checkpoint(pth_file, cfg=cfg, map_location=device)
ocr_model.decoder_device(device)

label_list = Path(base_dir).glob('*.txt')
test_dataset = None
for label_file in label_list:
    with label_file.open('r') as f:
        data_list = f.readlines()
    total_len = len(data_list)
    train_size = int(total_len * 0.8)
    valid_size = int(total_len * 0.1)
    random.shuffle(data_list)
    test_data = data_list[train_size+valid_size:]
    if not test_dataset:
        test_dataset = CustomImageDataset(base_dir=base_dir, img_dir=label_file.stem, 
                                        data_list = test_data, transform=None, 
                                        dataset_type='valid', regex=regex[label_file.stem], 
                                        lang_dict=lang_dict)
    else:
        test_dataset += CustomImageDataset(base_dir=base_dir, img_dir=label_file.stem, 
                                        data_list = test_data, transform=None, 
                                        dataset_type='valid', regex=regex[label_file.stem], 
                                        lang_dict=lang_dict)
results = {
    'C' : [],
    'J' : [],
    'K' : [],
    'E' : [],
    'M' : []
}

test_dataloader = DataLoader(test_dataset, batch_size=1, collate_fn=Custom_Collate(imgH=cfg.DATA.IMAGE_SIZE[0], imgW=cfg.DATA.IMAGE_SIZE[1]))

for item in test_dataloader:
    img, _, _, lang_num, text = item
    img = img.to(device)
    lang = lang_dict['id2token'][str(lang_num.item())]
    with torch.inference_mode():
        pred = ocr_model(img, lang)
    if text[0][0] == pred:
        results[lang].append(1)
    else:
        results[lang].append(0)

print()
for key in results:
    print(f'{key} : {(sum(results[key]) / len(results[key]))*100:.2f} %')
