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

# This script downloads the light field occlusions20 dataset for testing.
# DIR = "$( cd "$(dirname "$0")" ; pwd -P )"

echo "Downloading the sample light field data for evaluation ........"

wget --no-check-certificate https://github.com/monaen/LightFieldReconstruction/raw/storage/data/evaluation/occlusions20/occlusions_1.mat
wget --no-check-certificate https://github.com/monaen/LightFieldReconstruction/raw/storage/data/evaluation/occlusions20/occlusions_12.mat
wget --no-check-certificate https://github.com/monaen/LightFieldReconstruction/raw/storage/data/evaluation/occlusions20/occlusions_15.mat
wget --no-check-certificate https://github.com/monaen/LightFieldReconstruction/raw/storage/data/evaluation/occlusions20/occlusions_18.mat
wget --no-check-certificate https://github.com/monaen/LightFieldReconstruction/raw/storage/data/evaluation/occlusions20/occlusions_22.mat
wget --no-check-certificate https://github.com/monaen/LightFieldReconstruction/raw/storage/data/evaluation/occlusions20/occlusions_24.mat
wget --no-check-certificate https://github.com/monaen/LightFieldReconstruction/raw/storage/data/evaluation/occlusions20/occlusions_26.mat
wget --no-check-certificate https://github.com/monaen/LightFieldReconstruction/raw/storage/data/evaluation/occlusions20/occlusions_27.mat
wget --no-check-certificate https://github.com/monaen/LightFieldReconstruction/raw/storage/data/evaluation/occlusions20/occlusions_3.mat
wget --no-check-certificate https://github.com/monaen/LightFieldReconstruction/raw/storage/data/evaluation/occlusions20/occlusions_30.mat
wget --no-check-certificate https://github.com/monaen/LightFieldReconstruction/raw/storage/data/evaluation/occlusions20/occlusions_32.mat
wget --no-check-certificate https://github.com/monaen/LightFieldReconstruction/raw/storage/data/evaluation/occlusions20/occlusions_36.mat
wget --no-check-certificate https://github.com/monaen/LightFieldReconstruction/raw/storage/data/evaluation/occlusions20/occlusions_39.mat
wget --no-check-certificate https://github.com/monaen/LightFieldReconstruction/raw/storage/data/evaluation/occlusions20/occlusions_42.mat
wget --no-check-certificate https://github.com/monaen/LightFieldReconstruction/raw/storage/data/evaluation/occlusions20/occlusions_46.mat
wget --no-check-certificate https://github.com/monaen/LightFieldReconstruction/raw/storage/data/evaluation/occlusions20/occlusions_48.mat
wget --no-check-certificate https://github.com/monaen/LightFieldReconstruction/raw/storage/data/evaluation/occlusions20/occlusions_50.mat
wget --no-check-certificate https://github.com/monaen/LightFieldReconstruction/raw/storage/data/evaluation/occlusions20/occlusions_51.mat
wget --no-check-certificate https://github.com/monaen/LightFieldReconstruction/raw/storage/data/evaluation/occlusions20/occlusions_6.mat
wget --no-check-certificate https://github.com/monaen/LightFieldReconstruction/raw/storage/data/evaluation/occlusions20/occlusions_9.mat

mkdir testset/occlusions20
mv occlusions_1.mat testset/occlusions20/occlusions_1.mat
mv occlusions_12.mat testset/occlusions20/occlusions_12.mat
mv occlusions_15.mat testset/occlusions20/occlusions_15.mat
mv occlusions_18.mat testset/occlusions20/occlusions_18.mat
mv occlusions_22.mat testset/occlusions20/occlusions_22.mat
mv occlusions_24.mat testset/occlusions20/occlusions_24.mat
mv occlusions_26.mat testset/occlusions20/occlusions_26.mat
mv occlusions_27.mat testset/occlusions20/occlusions_27.mat
mv occlusions_3.mat testset/occlusions20/occlusions_3.mat
mv occlusions_30.mat testset/occlusions20/occlusions_30.mat
mv occlusions_32.mat testset/occlusions20/occlusions_32.mat
mv occlusions_36.mat testset/occlusions20/occlusions_36.mat
mv occlusions_39.mat testset/occlusions20/occlusions_39.mat
mv occlusions_42.mat testset/occlusions20/occlusions_42.mat
mv occlusions_46.mat testset/occlusions20/occlusions_46.mat
mv occlusions_48.mat testset/occlusions20/occlusions_48.mat
mv occlusions_51.mat testset/occlusions20/occlusions_50.mat
mv occlusions_50.mat testset/occlusions20/occlusions_51.mat
mv occlusions_6.mat testset/occlusions20/occlusions_6.mat
mv occlusions_9.mat testset/occlusions20/occlusions_9.mat