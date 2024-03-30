## Prepare datasets
It is recommended to symlink the dataset root to $CBN.PYTORCH/data. If your folder structure is different, you may need to change the corresponding paths in dataloader files.
```none
cbn.pytorch
└── data
    ├── CTW1500
    │   ├── train
    │   │   ├── text_image
    │   │   └── text_label_curve
    │   └── test
    │       ├── text_image
    │       └── text_label_curve
    ├── total_text
    │   ├── Images
    │   │   ├── Train
    │   │   └── Test
    │   └── Groundtruth
    │       ├── Polygon
    │       └── Rectangular
    ├── MSRA-TD500
    │   ├── train
    │   └── test
    ├── HUST-TR400
    ├── SynthText
    │   ├── 1
    │   ├── ...
    │   └── 200
```

## Download

These datasets can be downloaded from the following links:

- [CTW1500](https://github.com/Yuliang-Liu/Curve-Text-Detector), [train_images](https://universityofadelaide.box.com/shared/static/py5uwlfyyytbb2pxzq9czvu6fuqbjdh8.zip)[train_annos](https://universityofadelaide.box.com/shared/static/jikuazluzyj4lq6umzei7m2ppmt3afyw.zip) ,[test_images](https://universityofadelaide.box.com/shared/static/t4w48ofnqkdw7jyc4t11nsukoeqk9c3d.zip),[test_annos](https://cloudstor.aarnet.edu.au/plus/s/uoeFl0pCN9BOCN5) 

- [Total-Text](https://github.com/cs-chan/Total-Text-Dataset), [images](https://github.com/cs-chan/Total-Text-Dataset/tree/master/Dataset),[gts-.mat](https://drive.google.com/file/d/19quCaJGePvTc3yPZ7MAGNijjKfy77-ke/view?usp=sharing)

- MSRA-TD500 [[dataset]](http://www.iapr-tc11.org/dataset/MSRA-TD500/MSRA-TD500.zip)
- HUST-TR400 [[dataset]](http://mc.eistar.net/UpLoadFiles/dataset/HUST-TR400.zip)

- SynthText [[dataset]](https://www.robots.ox.ac.uk/~vgg/data/scenetext/)

- ICDAR2019-MLT [[dataset]](https://rrc.cvc.uab.es/?ch=15&com=downloads)

- ICDAR2017-MLT [[dataset]](https://rrc.cvc.uab.es/?ch=8&com=downloads)

- ICDAR2015 [[dataset]](https://rrc.cvc.uab.es/?ch=4&com=downloads)

- COCO-Text [[dataset]](https://rrc.cvc.uab.es/?ch=5&com=downloads)
