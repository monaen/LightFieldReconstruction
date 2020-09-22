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

# This script downloads the pretrained model of VGG19.
# DIR = "$( cd "$(dirname "$0")" ; pwd -P )"

echo "Downloading the pretrained model of VGG19 ........"

wget --no-check-certificate https://github.com/monaen/LightFieldReconstruction/raw/storage/vgg19/weights.part01.rar
wget --no-check-certificate https://github.com/monaen/LightFieldReconstruction/raw/storage/vgg19/weights.part02.rar
wget --no-check-certificate https://github.com/monaen/LightFieldReconstruction/raw/storage/vgg19/weights.part03.rar
wget --no-check-certificate https://github.com/monaen/LightFieldReconstruction/raw/storage/vgg19/weights.part04.rar
wget --no-check-certificate https://github.com/monaen/LightFieldReconstruction/raw/storage/vgg19/weights.part05.rar
wget --no-check-certificate https://github.com/monaen/LightFieldReconstruction/raw/storage/vgg19/weights.part06.rar
wget --no-check-certificate https://github.com/monaen/LightFieldReconstruction/raw/storage/vgg19/weights.part07.rar
wget --no-check-certificate https://github.com/monaen/LightFieldReconstruction/raw/storage/vgg19/weights.part08.rar
wget --no-check-certificate https://github.com/monaen/LightFieldReconstruction/raw/storage/vgg19/weights.part09.rar
wget --no-check-certificate https://github.com/monaen/LightFieldReconstruction/raw/storage/vgg19/weights.part10.rar
wget --no-check-certificate https://github.com/monaen/LightFieldReconstruction/raw/storage/vgg19/weights.part11.rar
wget --no-check-certificate https://github.com/monaen/LightFieldReconstruction/raw/storage/vgg19/weights.part12.rar
wget --no-check-certificate https://github.com/monaen/LightFieldReconstruction/raw/storage/vgg19/weights.part13.rar

echo "Unzipping ........"

unrar x weights.part01.rar && rm -f weights.part*.rar

echo "Done."

