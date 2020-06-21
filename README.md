# PVN3D
This is the source code for ***PVN3D: A Deep Point-wise 3D Keypoints Voting Network for 6DoF Pose Estimation***, **CVPR 2020**. ([PDF](https://arxiv.org/abs/1911.04231), [Video](https://www.bilibili.com/video/av89408773/)).

[![Watch the video](./pictures/video.jpg)](https://www.youtube.com/watch?v=ZKo788cyD-Q&t=1s)


## Installation
- Install CUDA9.0
- Set up python environment from requirement.txt:
  ```shell
  pip3 install -r requirement.txt 
  ```
- Install tkinter through ``sudo apt install python3-tk`` # python GUI工具
- Install [python-pcl](https://python-pcl-fork.readthedocs.io/en/latest/install.html#dependencies).
- Install PointNet++:
  ```shell
  python3 setup.py build_ext
  ```

## Datasets
- **LineMOD:** Download the preprocessed LineMOD dataset from [here](https://drive.google.com/drive/folders/19ivHpaKm9dOrr12fzC8IDFczWRPFxho7) (refer from [DenseFusion](https://github.com/j96w/DenseFusion)). Unzip it and link the unzipped ``Linemod_preprocessed/`` to ``pvn3d/datasets/linemod/Linemod_preprocessed``:
  ```shell
  ln -s path_to_unzipped_Linemod_preprocessed pvn3d/dataset/linemod/
  ```
- **YCB-Video:** Download the YCB-Video Dataset from [PoseCNN](https://rse-lab.cs.washington.edu/projects/posecnn/). Unzip it and link the unzipped```YCB_Video_Dataset``` to ```pvn3d/datasets/ycb/YCB_Video_Dataset```:

  ```shell
  ln -s path_to_unzipped_YCB_Video_Dataset pvn3d/datasets/ycb/
  ```


## Training and evaluating

### Training on the LineMOD Dataset
- First, generate synthesis data for each object using scripts from [raster triangle](https://github.com/ethnhe/raster_triangle).
- Train the model for the target object. Take object ape for example:
  ```shell
  cd pvn3d
  python3 -m train.train_linemod_pvn3d --cls ape
  ```
  The trained checkpoints are stored in ``train_log/linemod/checkpoints/{cls}/``, ``train_log/linemod/checkpoints/ape/`` in this example.

### Evaluating on the LineMOD Dataset
- Start evaluation by:
  ```shell
  cls='ape'
  tst_mdl=train_log/linemod/checkpoints/${cls}/${cls}_pvn3d_best.pth.tar
  python3 -m train.train_linemod_pvn3d -checkpoint $tst_mdl -eval_net --test --cls $cls
  ```
  You can evaluate different checkpoint by revising ``tst_mdl`` to the path of your target model.
- We provide our pre-trained models for each object [here](https://hkustconnect-my.sharepoint.com/personal/yhebk_connect_ust_hk/_layouts/15/onedrive.aspx?id=%2Fpersonal%2Fyhebk%5Fconnect%5Fust%5Fhk%2FDocuments%2FPVN3D%5Fpretrained%5Fmodel%2FLineMOD). Download them and move them to their according folders. For example, move the ``ape_pvn3d_best.pth.tar`` to ``train_log/linemod/checkpoints/ape/``. Then revise ``tst_mdl=train_log/linemod/checkpoints/ape/ape_pvn3d_best.path.tar`` for testing.

### Training on the YCB-Video Dataset
- Preprocess the validation set to speed up training:
  ```shell
  cd pvn3d
  python3 -m datasets.ycb.preprocess_testset
  ```
- Start training on the YCB-Video Dataset by:
  ```shell
  python3 -m train.train_ycb_pvn3d
  ```
  The trained model checkpoints are stored in ``train_log/ycb/checkpoints/``

### Evaluating on the YCB-Video Dataset
- Start evaluating by:
  ```shell
  tst_mdl=train_log/ycb/checkpoints/pvn3d_best.pth.tar
  python3 -m train.train_ycb_pvn3d -checkpoint $tst_mdl -eval_net --test
  ```
  You can evaluate different checkpoint by revising the ``tst_mdl`` to the path of your target model.
- We provide our pre-trained models [here](https://hkustconnect-my.sharepoint.com/:f:/g/personal/yhebk_connect_ust_hk/ElLgjzbbENZGhf-Sn8e4CMgBzd9zDjcJpCXFmB4n0WVw_w?e=IHMkvh). Download the ycb pre-trained model, move it to ``train_log/ycb/checkpoints/`` and modify ``tst_mdl`` for testing.

## Results
- Evaluation result on the LineMOD dataset:
  ![res_lm](pictures/res_linemod.png)
- Evaluation result on the YCB-Video dataset:
  ![res_ycb](pictures/res_ycb.png)
- Visualization of some predicted poses on YCB-Video dataset:
  ![vis_ycb](pictures/ycb_qualitive.png)
- Joint training for distinguishing objects with similar appearance but different in size:
  ![seg](pictures/seg_res.png)

## Citations
Please cite [PVN3D](https://arxiv.org/abs/1911.04231) if you use this repository in your publications:
```
@inproceedings{he2020pvn3d,
  title={PVN3D: A Deep Point-wise 3D Keypoints Voting Network for 6DoF Pose Estimation},
  author={He, Yisheng and Sun, Wei and Huang, haibin and Liu, Jianran and Fan, Haoqiang and Sun, Jian}
  booktitle={CVPR},
  year={2020}
}
```
