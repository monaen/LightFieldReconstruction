# High-Dimensional Dense Residual Convolutional Neural Network for Light Field Reconstruction [![MIT Licence](https://badges.frapsoft.com/os/mit/mit.svg?v=103)](https://opensource.org/licenses/mit-license.php)
This Project is a Tensorflow implementation of 
* "High-Dimensional Dense Residual Convolutional Neural Network for Light Field Reconstruction" *IEEE Transactions on Pattern Analysis and Machine Intelligence*, **Nan Meng**, Hayden K-H. So, Xing Sun, Edmund Y. Lam, 2019. [[Paper]](https://arxiv.org/pdf/1910.01426.pdf)
* "High-order Residual Network for Light Field Super-Resolution" *The 34th AAAI Conference on Artificial Intelligence*, **Nan Meng**, Xiaofei Wu, Jianzhuang Liu, Edmund Y. Lam, 2020. [[Paper]](https://arxiv.org/pdf/2003.13094.pdf)

This branch store the pretrained models for HDDRNet, M-HDDRNet and vgg. 

```
pretrained_models
  |--- HDDRNet
  |      |--- Sx2
  |      |--- Sx3
  |      |--- Sx4
  |      |--- Ax2
  |      |--- Ax3
  |      |--- Ax4
  |--- M-HDDRNet
  |      |--- Sx2
  |      |--- Sx3
  |      |--- Sx4
  |      |--- Ax2
  |      |--- Ax3
  |      |--- Ax4
  |---Others
  |      |--- A3x3-7x7
  |      |--- Sx2Ax2
  |--- README.md
vgg19
  |--- weights
data
  |--- evaluation
          |--- buddha.mat
  |--- testset
          |--- occlusions20
          |--- reflective20
```
