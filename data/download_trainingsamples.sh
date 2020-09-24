#!/usr/bin/env sh
# ====================================================================================================================== #
# | Description:                                                                                                       | #
# |     Script to download the training samples.                                                                       | #
# |                                                                                                                    | #
# |                                                                                                                    | #
# | Citation:                                                                                                          | #
# |     @article{meng2019high,                                                                                         | #
# |              title={High-dimensional dense residual convolutional neural network for light field reconstruction},  | #
# |              author={Meng, Nan and So, Hayden Kwok-Hay and Sun, Xing and Lam, Edmund},                             | #
# |              journal={IEEE Transactions on Pattern Analysis and Machine Intelligence},                             | #
# |              year={2019}                                                                                           | #
# |             }                                                                                                      | #
# |     @article{meng2020high,                                                                                         | #
# |              title={High-order residual network for light field super-resolution},                                 | #
# |              author={Meng, Nan and Wu, xiaofei and Liu, Jianzhuang and Lam, Edmund},                               | #
# |              journal={Association for the Advancement of Artificial Intelligence},                                 | #
# |              volume={34},                                                                                          | #
# |              number={7},                                                                                           | #
# |              pages={11757-11764},                                                                                  | #
# |              month={February},                                                                                     | #
# |              year={2020},                                                                                          | #
# |             }                                                                                                      | #
# |                                                                                                                    | #
# | Contact:                                                                                                           | #
# |     author: Nan Meng                                                                                               | #
# |     email:  u3003637@connect.hku.hk  |   nanmeng.uestc@hotmail.com                                                 | #
# ====================================================================================================================== #

# This script downloads the light field training samples.
# DIR = "$( cd "$(dirname "$0")" ; pwd -P )"

echo "Downloading the light field training samples ........"

wget --no-check-certificate https://github.com/monaen/LightFieldReconstruction/raw/storage/data/train/Angular/9x9/sample.npy
mkdir train/Angular/9x9
mv sample.npy train/Angular/9x9

wget --no-check-certificate https://github.com/monaen/LightFieldReconstruction/raw/storage/data/train/Spatial/5x5/sample.npy
mkdir train/Spatial/5x5
mv sample.npy train/Spatial/5x5

wget --no-check-certificate https://github.com/monaen/LightFieldReconstruction/raw/storage/data/train/SpatialAngular/5x5/sample.npy
mkdir train/SpatialAngular/5x5
mv sample.npy train/SpatialAngular/5x5