# Spatiotemporal-Untrammelled Mixture of Experts for Multi-Person Motion Prediction (ST-MoE)

Accepted by Association for the Advancement of Artificial Intelligence (AAAI2026-oral)
## Overview
Comprehensively and flexibly capturing the complex spatio
temporal dependencies of human motion is critical for multi
person motion prediction. Existing methods grapple with
 two primary limitations: i) Inflexible spatiotemporal repre
sentation due to reliance on positional encodings for cap
turing spatiotemporal information. ii) High computational
 costs stemming from the quadratic time complexity of con
ventional attention mechanisms. To overcome these limita
tions, we propose the Spatiotemporal-Untrammelled Mix
ture of Experts (ST-MoE), which flexibly explores complex
 spatio-temporal dependencies in human motion and signifi
cantly reduces computational cost. To adaptively mine com
plex spatio-temporal patterns from human motion, our model
 incorporates four distinct types of spatiotemporal experts,
 each specializing in capturing different spatial or temporal
 dependencies. To reduce the potential computational over
head while integrating multiple experts, we introduce bidi
rectional spatiotemporal Mamba as experts, each sharing
 bidirectional temporal and spatial Mamba in distinct com
binations to achieve model efficiency and parameter econ
omy. Extensive experiments on four multi-person benchmark
 datasets demonstrate that our approach not only outperforms
 state-of-art in accuracy but also reduces model parameter by
 41.38% and achieves a 3.6× speedup in training.

## Dataset and Environment
CMU-Mocap(UPMP) and Synthesized crowd datasets （Mix1 and Mix2）from [TBIFormer](https://github.com/xiaogangpeng/TBIFormer)

CHI3D from their [official website](https://ci3d.imar.ro/chi3d)

Experiments base on Cuda 11.6, torch1.13 and Python 3.8 （Perhaps not limited to these versions）
```
project_folder/
|-- Dataset
|   |-- CHI3D
|   |   `-- train
|   |-- Crowd
|   |   |-- mix1_6persons.npy
|   |   `-- mix2_10persons.npy
|   `-- Mocap
|       |-- test_3_75_mocap_umpm.npy
|       `-- train_3_75_mocap_umpm.npy
|-- Dataset_tools
|   |-- Dataset_CHI3D.py
|   `-- Dataset_Mocap.py
|-- main_CHI3D.py
|-- main_Mocap.py
|-- model
|   |-- checkpoint
|   |   |-- ...
|   |   `-- option.json
|   |-- Experts.py
|   |-- layers
|   |   |-- GCN.py
|   |   |-- Mamba.py
|   |   `-- TimeCon.py
|   `-- ST_MoE.py
|-- option
|   `-- option_Mocap.py
|-- requirements.txt
|-- structure.txt
|-- test_chi3d.sh
|-- test_ST_MoE.sh
|-- train_chi3d.sh
|-- train_ST_MoE.sh
`-- utils
    |-- data_utils.py
    |-- forward_kinematics.py
    `-- other_utils.py
```

## Train
To train on CMU_Mocap, simply run ``train_ST_MoE.sh``. For training on CHI3D, just execute ``train_chi3d.sh``. Don't forget to modify the paths in the ``Dataset_tools`` directory.

## Test
Checkpoints on CMU-Mocap(UMPM) and CHI3D can be download from [here](https://drive.google.com/drive/folders/1z5XpdMQAjdKktHe9LhLMo9iOf0BT4OvR?usp=sharing)

To run the tests, simply execute the corresponding shell script. Note that you need to modify the value of ``--ckpt`` in the option directory.

To switch datasets, you only need to change the value of the ``--dataset`` parameter.
## Citation


## Thanks
[TBIFormer](https://github.com/xiaogangpeng/TBIFormer)

[PGBIG](https://github.com/705062791/PGBIG)

[IAFormer](https://github.com/ArcticPole/IAFormer)
