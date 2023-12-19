import ray
import multiprocessing as mp
from pathlib import Path
from tqdm import tqdm
from PIL import Image, UnidentifiedImageError, ImageFile
import random
import json
import os
import shutil
import cv2
from ultralytics.data.converter import convert_coco


random.seed(0)

def split_dataset(json_list):
    total_len = len(json_list)

    train_size = int(total_len * 0.8)

    val_size = int(total_len * 0.1)

    train_data = json_list[:train_size]
    val_data = json_list[train_size:train_size + val_size]
    test_data = json_list[train_size + val_size:]

    return [train_data, 'train'], [val_data, 'valid'], [test_data, 'test']

num_process = mp.cpu_count()
ray.init(address='auto', dashboard_host="0.0.0.0")

@ray.remote
class IdGenerator(object):
    def __init__(self):
        self.image_id = -1
        self.annotation_id = -1

    def get_image_id(self):
        self.image_id += 1
        return self.image_id
    
    def get_annotation_id(self):
        self.annotation_id += 1
        return self.annotation_id


@ray.remote
class GlobalVars(object):
    def __init__(self):
        self.data = {'annotations': [],
                     'images': [],
                     'categories': [
                                        {
                                            "id": 1,
                                            "name": "Korean"
                                        },
                                        {
                                            "id": 2,
                                            "name": "Chinese"
                                        },
                                        {
                                            "id": 3,
                                            "name": "Japanese"
                                        },
                                        {
                                            "id": 4,
                                            "name": "English"
                                        },
                                        {
                                            "id": 5,
                                            "name": "Multi"
                                        }
                                    ]}
        self.cropped_data = {
            'K' : [],
            'C' : [],
            'J' : [],
            'E' : [],
            'M' : []
        }

    def get_data(self):
        return self.data
    
    def get_cropped_data(self):
        return self.cropped_data

    def update_data(self, image, annotation):
        self.data['images'].extend(image)
        self.data['annotations'].extend(annotation)

    def update_cropped_data(self, language, label):
        self.cropped_data[language].append(label)

@ray.remote
def json2coco(data_list, global_vars, id_generator, link_path, crop_image_path):
    link_path.mkdir(parents=True, exist_ok=True)
    local_image_list, local_anno_list = [], []
    token2categoryID = {
        'K' : 1,
        'C' : 2,
        'J' : 3,
        'E' : 4,
        'M' : 5
    }
    for json_file in tqdm(data_list, total=len(data_list)):
        try:
            image_id = ray.get(id_generator.get_image_id.remote())
            with json_file.open("r") as f:
                anno = json.load(f)

            image_source = str(json_file.parent / anno['Images']['file_name'])
            out_source = link_path / anno['Images']['file_name']
            if out_source.suffix == '.dng':
                out_source = out_source.with_suffix('.png')
            try:
                img = Image.open(image_source)
            except UnidentifiedImageError:
                img = Image.fromarray(cv2.cvtColor(cv2.imread(str(image_source)), cv2.COLOR_BGR2RGB))

            origin_w, origin_h = img.width, img.height
            ratio_x, ratio_y = 1920 / origin_w, 1080 / origin_h
            img = img.resize((1920, 1080))
    
            img.save(out_source)
            # shutil.copy(image_source, out_source)
            image = {
                "id" : image_id,
                "file_name" : anno['Images']['file_name'],
                "width": anno["Images"]["width"] * ratio_x,
                "height": anno["Images"]["height"] * ratio_y
            }

            # link_name = link_path / anno['Images']['file_name']
            # os.symlink(image_source, link_name)
            for bbox in anno["Annotation"]:
                try:
                    anno_id = ray.get(id_generator.get_annotation_id.remote())
                    x_coords = [point[0] * ratio_x for point in bbox['polygon_points']]
                    y_coords = [point[1] * ratio_y for point in bbox['polygon_points']]
                    bounding_box = [min(x_coords), min(y_coords), max(x_coords), max(y_coords)]

                    cropped_image = img.crop(bounding_box)

                    annotation = {
                        "id" : anno_id,
                        "image_id" : image_id,
                        "category_id" : token2categoryID[bbox['text_language']],
                        "segmentation" : [[j for i in zip(x_coords, y_coords) for j in i]],
                        "bbox" : bounding_box,
                        "text" : bbox['text'],
                        "iscrowd": 0
                    }
                    crop_dest_path = crop_image_path / f'{bbox["text_language"]}'
                    crop_dest_path.mkdir(parents=True, exist_ok=True)

                    cropped_image.save(crop_dest_path / f'{anno_id}.png')
                    label = f'{anno_id}.png\t{bbox["text"]}'
                    global_vars.update_cropped_data.remote(bbox['text_language'], label)

                    local_anno_list.append(annotation)
                except:
                    continue
            local_image_list.append(image)
        except:
            continue

        

    global_vars.update_data.remote(local_image_list, local_anno_list)



base = Path('Path') # Input Your File Path
link_path = Path('coco_converted/images')
shutil.rmtree(link_path.parent)

crop_image_path = Path('cropped_image')
shutil.rmtree(crop_image_path)
crop_image_path.mkdir(parents=True, exist_ok=True)

# Load Json File Path
json_list = []

for p1 in base.glob('*'):
    if not p1.is_dir():
        continue
    for p2 in p1.glob('*'):
        if not p2.is_dir():
            continue
        json_list.extend(list(p2.glob('*.json')))

random.shuffle(json_list)

for data_list, file_name in split_dataset(json_list):
    global_vars = GlobalVars.remote()
    id_generator = IdGenerator.remote()
    step = len(data_list) // num_process + 1
    tasks = [json2coco.remote(data_list[step*x:step*(x+1)],global_vars,id_generator, link_path / file_name, crop_image_path) for x in range(num_process)]
    ray.get(tasks)
    results = ray.get(global_vars.get_data.remote())
    with open(file_name + '.json', 'w') as f:
        json.dump(results, f)

    crop_results = ray.get(global_vars.get_cropped_data.remote())
    for key in crop_results:
        with open(crop_image_path / f'{key}.txt', 'w') as f:
            f.write('\n'.join(crop_results[key]))

out_path = Path("annotations")
if out_path.exists():
    shutil.rmtree(out_path)

convert_coco(labels_dir='./', save_dir=out_path, use_segments=True)

shutil.move(out_path / 'labels', Path('./coco_converted/'))
shutil.rmtree(out_path)