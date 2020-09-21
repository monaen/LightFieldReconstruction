#!/usr/bin/env sh
# ====================================================================================================================== #
# | Description:                                                                                                       | #
# |     Script to download the pretrained models                                                                       | #
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

# This script downloads the pretrained model of M-HDDRNet for Sx4 task and unzips it.
# DIR = "$( cd "$(dirname "$0")" ; pwd -P )"

echo "Downloading the pretrained model of M-HDDRNet for Sx4 task ........"

wget --no-check-certificate https://github.com/monaen/LightFieldReconstruction/raw/storage/pretrained_models/M-HDDRNet/Sx4/M-HDDRNet_Sx4.part01.rar
wget --no-check-certificate https://github.com/monaen/LightFieldReconstruction/raw/storage/pretrained_models/M-HDDRNet/Sx4/M-HDDRNet_Sx4.part02.rar
wget --no-check-certificate https://github.com/monaen/LightFieldReconstruction/raw/storage/pretrained_models/M-HDDRNet/Sx4/M-HDDRNet_Sx4.part03.rar
wget --no-check-certificate https://github.com/monaen/LightFieldReconstruction/raw/storage/pretrained_models/M-HDDRNet/Sx4/M-HDDRNet_Sx4.part04.rar
wget --no-check-certificate https://github.com/monaen/LightFieldReconstruction/raw/storage/pretrained_models/M-HDDRNet/Sx4/M-HDDRNet_Sx4.part05.rar
wget --no-check-certificate https://github.com/monaen/LightFieldReconstruction/raw/storage/pretrained_models/M-HDDRNet/Sx4/M-HDDRNet_Sx4.part06.rar
wget --no-check-certificate https://github.com/monaen/LightFieldReconstruction/raw/storage/pretrained_models/M-HDDRNet/Sx4/M-HDDRNet_Sx4.part07.rar
wget --no-check-certificate https://github.com/monaen/LightFieldReconstruction/raw/storage/pretrained_models/M-HDDRNet/Sx4/M-HDDRNet_Sx4.part08.rar
wget --no-check-certificate https://github.com/monaen/LightFieldReconstruction/raw/storage/pretrained_models/M-HDDRNet/Sx4/M-HDDRNet_Sx4.part09.rar
wget --no-check-certificate https://github.com/monaen/LightFieldReconstruction/raw/storage/pretrained_models/M-HDDRNet/Sx4/M-HDDRNet_Sx4.part10.rar

echo "Unzipping ........"

unrar x M-HDDRNet_Sx4.part01.rar && rm -f M-HDDRNet_Sx4.part*.rar

echo "Done."

