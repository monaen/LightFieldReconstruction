# High-Dimensional Dense Residual Convolutional Neural Network for Light Field Reconstruction [![MIT Licence](https://badges.frapsoft.com/os/mit/mit.svg?v=103)](https://opensource.org/licenses/mit-license.php)

**Note: The code for this project will be uploaded gradually.**

This Project is a Tensorflow implementation of 
* "High-Dimensional Dense Residual Convolutional Neural Network for Light Field Reconstruction" *IEEE Transactions on Pattern Analysis and Machine Intelligence*, **Nan Meng**, Hayden K-H. So, Xing Sun, Edmund Y. Lam, 2019. [[Paper]](https://arxiv.org/pdf/1910.01426.pdf)
* "High-order Residual Network for Light Field Super-Resolution" *The 34th AAAI Conference on Artificial Intelligence*, **Nan Meng**, Xiaofei Wu, Jianzhuang Liu, Edmund Y. Lam, 2020. [[Paper]](https://arxiv.org/pdf/2003.13094.pdf)


## Requirements
* Python2==2.7
* Python3>=3.5
* Tensorflow r1.\*>=r1.8
* tqmd
* opencv
* unrar

## Installation
* Download the project
```commandline
git clone https://github.com/monaen/LightFieldReconstruction.git --branch master --single-branch
```

* Preparing the Dataset
```commandline
```

## Training
#### Training models for spatial super-resolution tasks (Sx2, Sx3, and Sx4)
* Training the model for spatial super-resolution (e.g. Sx4)
```commandline
python train_SpatialSR.py --gamma_S 4 --verbose --perceptual_loss
```


```commandline
# usage: train_SpatialSR.py [-h] [--datadir DATADIR] [--lr_start LR_START]
#                           [--lr_beta1 LR_BETA1] [--batchSize BATCHSIZE]
#                           [--imageSize IMAGESIZE] [--viewSize VIEWSIZE]
#                           [--channels CHANNELS] [--verbose VERBOSE]
#                           [--num_epoch NUM_EPOCH] [--start_epoch START_EPOCH]
#                           [--gamma_S {1,2,3,4}] [--gamma_A {0,1,2,3,4}]
#                           [--num_GRL_HRB NUM_GRL_HRB]
#                           [--num_SRe_HRB NUM_SRE_HRB] [--resume RESUME]
#                           [--select_gpu SELECT_GPU]
#                           [--perceptual_loss PERCEPTUAL_LOSS]
#                           [--vgg_model VGG_MODEL] [--save_folder SAVE_FOLDER]
```

## Evaluation
```commandline
```

## Using the pretrained model
To download the pretrained mdoels, please change the directory into the folder `pretrained_models` and run the corresponding bash files. For example, to download the HDDRNet_Sx4 pretrained model, 
```commandline
# path = Path to LightFieldReconstruction
cd pretrained_models
bash download_pretrained_models_HDDRNet_Sx4.sh
```


## Reference
Paper
```
@article{Meng2019High,
  title        = {High-dimensional dense residual convolutional neural network for light field reconstruction},
  author       = {Meng, Nan and So, Hayden Kwok-Hay and Sun, Xing and Lam, Edmund},
  journal      = {IEEE Transactions on Pattern Analysis and Machine Intelligence},
  year         = {2019}
}
@article{Meng2020High,
  title        = {High-order residual network for light field super-resolution},
  author       = {Meng, Nan and Wu, Xiaofei and Liu, Jianzhuang and Lam, Edmund},
  journal      = {Thirty-Fourth AAAI Conference on Artificial Intelligence},
  year         = {2020}
}
```
## Acknowledgement
Our code is released under [MIT License](https://github.com/monaen/LightFieldReconstruction/blob/master/LICENSE). We would like to appreciate the GPU support from **Lee Ultrasound Imaging Group** of [Prof.Wei-Ning Lee](https://www.eee.hku.hk/~wnlee/people.html)

## TODO List
* T-PAMI
- [x] Spatial super-resolution 2x framework.
- [x] Spatial super-resolution 3x framework.
- [x] Spatial super-resolution 4x framework.
- [x] Angular super-resolution 2x (5x5->9x9) framework.
- [x] Angular super-resolution 2.3x (3x3->7x7) framework.
- [x] Angular super-resolution 3x (3x3->9x9) framework.
- [x] Angular super-resolution 4x (2x2->8x8) framework.
- [x] Spatial super-resolution 2x Angular super-resolution 2x framework.
- [x] The pretrained models for spatial Sx2, Sx3, Sx4.
- [ ] Training data.
- [ ] Testing data.
- [ ] Analysis preparation.
* AAAI
- [ ] Spatial super-resolution 2x framework and the pretrained model.
- [ ] Spatial super-resolution 3x framework and the pretrained model.
- [ ] Spatial super-resolution 4x framework and the pretrained model.
- [ ] Angular super-resolution 2x (5x5->9x9) framework and the pretrained model.
- [ ] Angular super-resolution 2.3x (3x3->7x7) framework and the pretrained model.
- [ ] Angular super-resolution 3x (3x3->9x9) framework and the pretrained model.
- [ ] Angular super-resolution 4x (2x2->8x8) framework and the pretrained model.
- [ ] Other materials.
