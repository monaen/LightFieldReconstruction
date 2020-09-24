#!/usr/bin/env sh
# ====================================================================================================================== #
# | Description:                                                                                                       | #
# |     Script to download the evaluation data (reflective 20).                                                        | #
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

# This script downloads the light field reflective20 dataset for testing.
# DIR = "$( cd "$(dirname "$0")" ; pwd -P )"

echo "Downloading the sample light field data for evaluation ........"

wget --no-check-certificate https://github.com/monaen/LightFieldReconstruction/raw/storage/data/testset/reflective20/reflective_1.mat
wget --no-check-certificate https://github.com/monaen/LightFieldReconstruction/raw/storage/data/testset/reflective20/reflective_3.mat
wget --no-check-certificate https://github.com/monaen/LightFieldReconstruction/raw/storage/data/testset/reflective20/reflective_5.mat
wget --no-check-certificate https://github.com/monaen/LightFieldReconstruction/raw/storage/data/testset/reflective20/reflective_6.mat
wget --no-check-certificate https://github.com/monaen/LightFieldReconstruction/raw/storage/data/testset/reflective20/reflective_8.mat
wget --no-check-certificate https://github.com/monaen/LightFieldReconstruction/raw/storage/data/testset/reflective20/reflective_9.mat
wget --no-check-certificate https://github.com/monaen/LightFieldReconstruction/raw/storage/data/testset/reflective20/reflective_10.mat
wget --no-check-certificate https://github.com/monaen/LightFieldReconstruction/raw/storage/data/testset/reflective20/reflective_12.mat
wget --no-check-certificate https://github.com/monaen/LightFieldReconstruction/raw/storage/data/testset/reflective20/reflective_15.mat
wget --no-check-certificate https://github.com/monaen/LightFieldReconstruction/raw/storage/data/testset/reflective20/reflective_16.mat
wget --no-check-certificate https://github.com/monaen/LightFieldReconstruction/raw/storage/data/testset/reflective20/reflective_18.mat
wget --no-check-certificate https://github.com/monaen/LightFieldReconstruction/raw/storage/data/testset/reflective20/reflective_20.mat
wget --no-check-certificate https://github.com/monaen/LightFieldReconstruction/raw/storage/data/testset/reflective20/reflective_22.mat
wget --no-check-certificate https://github.com/monaen/LightFieldReconstruction/raw/storage/data/testset/reflective20/reflective_24.mat
wget --no-check-certificate https://github.com/monaen/LightFieldReconstruction/raw/storage/data/testset/reflective20/reflective_25.mat
wget --no-check-certificate https://github.com/monaen/LightFieldReconstruction/raw/storage/data/testset/reflective20/reflective_26.mat
wget --no-check-certificate https://github.com/monaen/LightFieldReconstruction/raw/storage/data/testset/reflective20/reflective_27.mat
wget --no-check-certificate https://github.com/monaen/LightFieldReconstruction/raw/storage/data/testset/reflective20/reflective_28.mat
wget --no-check-certificate https://github.com/monaen/LightFieldReconstruction/raw/storage/data/testset/reflective20/reflective_30.mat
wget --no-check-certificate https://github.com/monaen/LightFieldReconstruction/raw/storage/data/testset/reflective20/reflective_32.mat

mkdir testset/reflective20
mv reflective_1.mat testset/reflective20/reflective_1.mat
mv reflective_3.mat testset/reflective20/reflective_3.mat
mv reflective_5.mat testset/reflective20/reflective_5.mat
mv reflective_6.mat testset/reflective20/reflective_6.mat
mv reflective_8.mat testset/reflective20/reflective_8.mat
mv reflective_9.mat testset/reflective20/reflective_9.mat
mv reflective_10.mat testset/reflective20/reflective_10.mat
mv reflective_12.mat testset/reflective20/reflective_12.mat
mv reflective_15.mat testset/reflective20/reflective_15.mat
mv reflective_16.mat testset/reflective20/reflective_16.mat
mv reflective_18.mat testset/reflective20/reflective_18.mat
mv reflective_20.mat testset/reflective20/reflective_20.mat
mv reflective_22.mat testset/reflective20/reflective_22.mat
mv reflective_24.mat testset/reflective20/reflective_24.mat
mv reflective_25.mat testset/reflective20/reflective_25.mat
mv reflective_26.mat testset/reflective20/reflective_26.mat
mv reflective_27.mat testset/reflective20/reflective_27.mat
mv reflective_28.mat testset/reflective20/reflective_28.mat
mv reflective_30.mat testset/reflective20/reflective_30.mat
mv reflective_32.mat testset/reflective20/reflective_32.mat