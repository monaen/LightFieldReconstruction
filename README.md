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
```commandline
```

## Evaluation
```commandline
```

## Using the pretrained model
```commandline
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

## TODO List
* T-PAMI
- [x] Spatial super-resolution 2x framework.
- [x] Spatial super-resolution 3x framework.
- [x] Spatial super-resolution 4x framework.
- [ ] Angular super-resolution 2x (5x5->9x9) framework.
- [ ] Angular super-resolution 3x (3x3->9x9) framework.
- [ ] Angular super-resolution 4x (2x2->8x8) framework.
- [ ] The pretrained models.
- [ ] Training data.
- [ ] Testing data.
- [ ] Analysis preparation.
* AAAI
- [ ] Spatial super-resolution 2x framework and the pretrained model.
- [ ] Spatial super-resolution 3x framework and the pretrained model.
- [ ] Spatial super-resolution 4x framework and the pretrained model.
- [ ] Angular super-resolution 2x (5x5->9x9) framework and the pretrained model.
- [ ] Angular super-resolution 3x (3x3->9x9) framework and the pretrained model.
- [ ] Angular super-resolution 4x (2x2->8x8) framework and the pretrained model.
- [ ] Other materials.
