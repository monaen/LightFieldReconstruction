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

* Preparing the Dataset (preparing)
```commandline
```

## Training
#### Training models for spatial or angular or both super-resolution tasks
* Training the model for spatial super-resolution (e.g. Sx4). You need to specify the `gamma_S` for different spatial SR tasks.
```commandline
python train_SpatialSR.py --gamma_S 4 --verbose --perceptual_loss
```
* Training the model for angular super-resolution (e.g. Ax4). You need to specify the `gamma_A` for different angular SR tasks.
```commandline
python train_ViewSysthesis.py --gamma_A 4 --verbose
```
* Training the model for spatio-angular super-resolution (e.g. Sx2Ax2). You need to specify both the `gamma_S` and `gamma_A` for different spatio-angular SR tasks.
```commandline
python train_ViewSysthesis.py --gamma_S 2 --gamma_A 2 --verbose --perceptual_loss
```

### Training arguments
```commandline
usage:train_SpatialSR.py [-h] [--datadir DATADIR] [--lr_start LR_START] [--lr_beta1 LR_BETA1] [--batchSize BATCHSIZE]
                         [--imageSize IMAGESIZE] [--viewSize VIEWSIZE] [--channels CHANNELS] [--verbose VERBOSE]
                         [--num_epoch NUM_EPOCH] [--start_epoch START_EPOCH] [--gamma_S {1,2,3,4}]
                         [--gamma_A {0,1,2,3,4}] [--num_GRL_HRB NUM_GRL_HRB] [--num_SRe_HRB NUM_SRE_HRB]
                         [--resume RESUME] [--select_gpu SELECT_GPU] [--perceptual_loss PERCEPTUAL_LOSS]
                         [--vgg_model VGG_MODEL] [--save_folder SAVE_FOLDER]

optional arguments:
  -h, --help                Show this help message and exit
  --datadir                 The training and testing data path
  --lr_start                The start learning rate
  --lr_beta1                The exponential decay rate for the 1st moment estimates
  --batchSize               The batchsize of the input data
  --imageSize               Spatial size of the input light fields
  --viewSize                Angular size of the input light fields
  --channels                Channels=1 means only the luma channel; Channels=3 means RGB channels (not supported)
  --verbose                 Whether print the network structure or not
  --num_epoch               The total number of training epoch
  --start_epoch             The start epoch counting number
  --gamma_S {1,2,3,4}       Spatial downscaling factor
  --gamma_A {0,1,2,3,4}     Angular downscaling factor, '0' represents 3x3->7x7
  --num_GRL_HRB             The number of HRB in GRLNet (only for AAAI model)
  --num_SRe_HRB             The number of HRB in SReNet (only for AAAI model)
  --resume                  Need to resume the pretrained model or not
  --select_gpu              Select the gpu for training or evaluation
  --perceptual_loss         Need to use perceptual loss or not, if true, one also have to set the vgg_model item
  --vgg_model               Pretrained VGG model path
  --save_folder             Model save path
```

## Evaluation
* Spatial SR evaluation (Sx4, Sx3, Sx2)
```commandline
python evaluation_SpatialSR.py --datapath $PATH_TO_THE_EVALUATION_LF_DATA --gamma_S $Upscaling_Factor --pretrained_model $PATH_TO_THE_PRETRAINED_MODEL --select_gpu 0 --verbose
```
Here, we provide an example to evaluate the performance on Sx4 task as guidance. 
```commandline
# change to the root folder 'LightFieldReconstruction' of the project
cd data
bash download_evaluation_data.sh
cd ../pretrained_models
bash download_pretrained_models_HDDRNet_Sx4.sh
cd ..

python evaluation_SpatialSR.py --datapath data/evaluation/buddha.mat --gamma_S 4 --pretrained_model pretrained_models/HDDRNet/Sx4/HDDRNet --select_gpu 0 --verbose
```
* Angular SR evaluation (Ax4, Ax3, Ax2, A3x3_7x7)  [Preparing]

## Using the pretrained model
To download the pretrained mdoels, please change the directory into the folder `pretrained_models` and run the corresponding bash files. For example, to download the HDDRNet_Sx4 pretrained model, 
```commandline
# path = Path to LightFieldReconstruction
cd pretrained_models
bash download_pretrained_models_HDDRNet_Sx4.sh
```
We provide a detailed [instruction](https://github.com/monaen/LightFieldReconstruction/tree/master/pretrained_models) on how to download the pretrained models for differnet SR models and tasks.

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

## Frequently Asked Questions

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
- [ ] The pretrained models for angular Ax2, Ax2.3, Ax3, Ax4.
- [ ] The pretrained models for spatio-angular Sx2Ax2.
- [ ] Training data.
- [ ] Testing data.
- [ ] Evaluation code.
- [ ] Results and analysis.
* AAAI
- [ ] Spatial super-resolution 2x framework and the pretrained model.
- [ ] Spatial super-resolution 3x framework and the pretrained model.
- [ ] Spatial super-resolution 4x framework and the pretrained model.
- [ ] Angular super-resolution 2x (5x5->9x9) framework and the pretrained model.
- [ ] Angular super-resolution 2.3x (3x3->7x7) framework and the pretrained model.
- [ ] Angular super-resolution 3x (3x3->9x9) framework and the pretrained model.
- [ ] Angular super-resolution 4x (2x2->8x8) framework and the pretrained model.
- [ ] Other materials.

## Acknowledgement
Our code is released under [MIT License](https://github.com/monaen/LightFieldReconstruction/blob/master/LICENSE). We would like to appreciate the GPU support from **Lee Ultrasound Imaging Group** of [Prof.Wei-Ning Lee](https://www.eee.hku.hk/~wnlee/people.html)
