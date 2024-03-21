# [CBNet: A Plug-and-Play Network for Segmentation-based Scene Text Detection](https://link.springer.com/article/10.1007/s11263-024-02022-w)

```
@article{zhao2024cbnet,
  title={CBNet: A Plug-and-Play Network for Segmentation-based Scene Text Detection},
  author={Zhao, Xi and Feng, Wei and Zhang, Zheng and Lv, Jingjing and Zhu, Xin and Lin, Zhangang and Hu, Jinghe and Shao, Jingping},
  journal={International Journal of Computer Vision},
  pages={1--20},
  year={2024},
  publisher={Springer}
}
```

## Results and Models
- CTW1500

| Method | Backbone | Finetune | Precision (%) | Recall (%) | F-measure (%) | Config |
| :-: | :-: | :-: | :-: | :-: | :-: | :-: |
| CBN | ResNet18 | N | 87.2 | 79.8 | 83.3 | [config](cbn_r18_ctw.py) | - |
| [RSCA](https://openaccess.thecvf.com/content/CVPR2021W/MAI/papers/Li_RSCA_Real-Time_Segmentation-Based_Context-Aware_Scene_Text_Detection_CVPRW_2021_paper.pdf) | Mobilenetv3 | N | 81.7 | 73.8 | 77.5 | - |
| CBN | MobileNetv3 | N | 88.7 | 73.2 | 80.2 | [config](cbn_mobilev3_ctw.py) | - |
| CBN | ResNet18 | Y(MLT19) | 89.3 | 82.9 | 86.0 | [config](cbn_r18_ctw_finetune_ic19.py) | - |

- Total-Text

| Method | Backbone | Finetune | Precision (%) | Recall (%) | F-measure (%) | Config |
| :-: | :-: | :-: | :-: | :-: | :-: | :-: |
| CBN | ResNet18 | N | 90.1 | 79.1 | 84.5 | [config](cbn_r18_tt.py) |
| CBN | MobileNetv3 | N | 87.6 | 78.1 | 82.5 | [config](cbn_mobilev3_tt.py) |
| CBN | ResNet18 | Y(MLT19) | 89.3 | 85.2 | 87.2 | [config](cbn_r18_tt_finetune_ic19.py) |

- MSRA-TD500

| Method | Backbone | Finetune | Precision (%) | Recall (%) | F-measure (%) | Config |
| :-: | :-: | :-: | :-: | :-: | :-: | :-: |
| CBN | ResNet18 | N | 83.8 | 80.8 | 82.3 | [config](cbn_r18_msra.py) |
| CBN | MobileNetv3 | N | 82.3 | 79.1 | 80.6 | [config](cbn_mobilev3_msra.py) |
| CBN | ResNet18 | Y(MLT19) | 93.2 | 86.4 | 89.7 | [config](cbn_r18_msra_finetune_ic19.py) |

- ICDAR2015

| Method | Backbone | Finetune | Scale | Precision (%) | Recall (%) | F-measure (%) | Config |
| :-: | :-: | :-: | :-: | :-: | :-: | :-: | :-: |
| EAST | MobileNetV3 | N | L:2400 | 78.20 | 79.10 | 78.65 | - |
| PSE | MobileNetV3 | N | S:736 | 82.20 | 70.48 | 75.89 | - |
| DB | MobileNetV3 | N | S:736 | 77.29 | 73.08 | 75.12 | - |
| PAN | MobileNetV3 | N | S:736 | 84.2 | 75.2 | 79.4 | - |
| PAN | MobileNetV3 | N | S:1024 | 82.6 | 77.4 | 79.9 | - |
| CBN | MobileNetV3 | N | S:736 | 84.1 | 75.5 | 79.6 | [config](cbn_mobilev3_ic15.py) |
| CBN | MobileNetV3 | N | S:1024 | 82.4 | 78.6 | 80.5 | [config](cbn_mobilev3_ic15.py) |


- SynthText

| Method | Backbone |           Config           |
| :----: | :------: | :------------------------: |
|  CBN   | ResNet18 | [config](cbn_r18_synth.py) |
|  CBN   | MobileNetv3| [config](cbn_mobilev3_synth.py) |

