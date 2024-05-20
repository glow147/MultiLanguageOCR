# MultiLanguageOCR

This code is tried to base on <a href=https://arxiv.org/pdf/2103.15992.pdf>[A Multiplexed Network for End-to-End, Multilingual OCR]</a>

Only take multi language recognition method ( shared feature extract networks ), 
this means that is not a end-to-end model and don't shared results of detection / recognition

Also, Yolov8 model (ultralytics) on Detection, Resnet18(Feature Extractor, Encoder) + Transformer(Decoder) on Recognition were used.

Dataset is not opened yet. (2024-05-20)


## Method

Firstly, do preprocess dataset for train yolo and multi language recognition model.
```
ray start --head
python split_data.py
```

Secondly, train detection model.
```
python yolo.py
```

Thirdly, train recognition model using lightning module.
```
python lightning_main.py
```

Lastly, test recognition model using lightning module.
```
python test.py

Recognition Results(Exact Match) :  
C : 86.97 %  
J : 49.79 %  
K : 73.54 %  
E : 80.38 %  
M : 68.17 %  
```

You can see how to use this model in demo.ipynb
