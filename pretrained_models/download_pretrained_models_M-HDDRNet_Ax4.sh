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

# This script downloads the pretrained model of M-HDDRNet for Ax4 task and unzips it.
# DIR = "$( cd "$(dirname "$0")" ; pwd -P )"

echo "Downloading the pretrained model of M-HDDRNet for Ax4 task ........"

wget --no-check-certificate https://github.com/monaen/LightFieldReconstruction/raw/storage/pretrained_models/M-HDDRNet/Ax4/M-HDDRNet_Ax4.part1.rar
wget --no-check-certificate https://github.com/monaen/LightFieldReconstruction/raw/storage/pretrained_models/M-HDDRNet/Ax4/M-HDDRNet_Ax4.part2.rar
wget --no-check-certificate https://github.com/monaen/LightFieldReconstruction/raw/storage/pretrained_models/M-HDDRNet/Ax4/M-HDDRNet_Ax4.part3.rar


echo "Unzipping ........"

unrar x M-HDDRNet_Ax4.part1.rar && rm -f M-HDDRNet_Ax4.part*.rar

echo "Done."

