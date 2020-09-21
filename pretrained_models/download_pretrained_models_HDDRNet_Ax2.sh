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

# This script downloads the pretrained model of HDDRNet for Sx2 task and unzips it.
# DIR = "$( cd "$(dirname "$0")" ; pwd -P )"

echo "Downloading the pretrained model of HDDRNet for Ax2 task ........"

wget --no-check-certificate https://github.com/monaen/LightFieldReconstruction/raw/storage/pretrained_models/HDDRNet/Ax2/HDDRNet_Ax2.part1.rar
wget --no-check-certificate https://github.com/monaen/LightFieldReconstruction/raw/storage/pretrained_models/HDDRNet/Ax2/HDDRNet_Ax2.part2.rar
wget --no-check-certificate https://github.com/monaen/LightFieldReconstruction/raw/storage/pretrained_models/HDDRNet/Ax2/HDDRNet_Ax2.part3.rar


echo "Unzipping ........"

unrar x HDDRNet_Ax2.part1.rar && rm -f HDDRNet_Ax2.part*.rar

echo "Done."

