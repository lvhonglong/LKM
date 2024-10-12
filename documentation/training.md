# Training Documentation

This documentation only shows the way to re-produce our paper. If you would like to remove or add a dataset to the training, you are responsible for adapting the training code yourself.

## Datasets

The following datasets are used during our training.

**IMPORTANT:Our training strategy refers to [RVM](https://github.com/PeterL1n/RobustVideoMatting/blob/master/documentation/training.md), and the dataset configuration is similar to it. We've modified the background training set for image training and added two new test sets. Since the Distinctions-646 and Adobe Image Matting datasets have not been made public, we did not test on these two datasets.**

### Matting Datasets
* [VideoMatte240K](https://grail.cs.washington.edu/projects/background-matting-v2/#/datasets)
    * Download JPEG SD version (6G) for stage 1 and 2.
    * Download JPEG HD version (60G) for stage 3 and 4.
    * Manually move clips `0000`, `0100`, `0200`, `0300` from the training set to a validation set.
* ImageMatte
    * ImageMatte consists of [Distinctions-646](https://wukaoliu.github.io/HAttMatting/) and [Adobe Image Matting](https://sites.google.com/view/deepimagematting) datasets.
    * Only needed for stage 4.
    * You need to contact their authors to acquire.
    * After downloading both datasets, merge their samples together to form ImageMatte dataset.
    * Only keep samples of humans.
    * Full list of images we used in ImageMatte for training:
        * [imagematte_train.txt](/documentation/misc/imagematte_train.txt)
        * [imagematte_valid.txt](/documentation/misc/imagematte_valid.txt)
### Background Datasets
* Video Backgrounds
    * You can download the preprocessed versions:
        * [Train set (14.6G)](https://robustvideomatting.blob.core.windows.net/data/BackgroundVideosTrain.tar) (Manually move some clips to validation set)
* Image Backgrounds
    * We use [BG-2K](https://github.com/JizhiziLi/GFM?tab=readme-ov-file#bg-20k). It contains 20,000 high-resolution background images excluded salient objects, which can be used to help generate high quality synthetic data.

### Segmentation Datasets

* [COCO](https://cocodataset.org/#download)
    * Download [train2017.zip (18G)](http://images.cocodataset.org/zips/train2017.zip)
    * Download [panoptic_annotations_trainval2017.zip (821M)](http://images.cocodataset.org/annotations/panoptic_annotations_trainval2017.zip)
    * Note that our train script expects the panopitc version.
* [YouTubeVIS 2021](https://youtube-vos.org/dataset/vis/)
    * Download the train set. No preprocessing needed.
* [Supervisely Person Dataset](https://supervise.ly/explore/projects/supervisely-person-dataset-23304/datasets)
    * you can download the [preprocessed version (800M)](https://robustvideomatting.blob.core.windows.net/data/SuperviselyPersonDataset.tar)

## Training

For reference, our training was done on 2 Nvidia 3090 24G GPUs.

After you have downloaded the datasets. Please configure `train_config.py` to provide paths to your datasets.

The training consists of 4 stages. For detail, please refer to the [paper](https://peterl1n.github.io/RobustVideoMatting/).

### Stage 1
```sh
python train.py \
    --model-variant unireplknet \
    --dataset videomatte \
    --resolution-lr 512 \
    --seq-length-lr 15 \
    --learning-rate-backbone 0.0001 \
    --learning-rate-aspp 0.0002 \
    --learning-rate-decoder 0.0002 \
    --learning-rate-refiner 0 \
    --checkpoint-dir checkpoint/stage1 \
    --log-dir log/stage1 \
    --epoch-start 0 \
    --epoch-end 15
```

### Stage 2
```sh
python train.py \
    --model-variant unireplknet \
    --dataset videomatte \
    --resolution-lr 512 \
    --seq-length-lr 20 \
    --learning-rate-backbone 0.00005 \
    --learning-rate-aspp 0.0001 \
    --learning-rate-decoder 0.0001 \
    --learning-rate-refiner 0 \
    --checkpoint checkpoint/stage1/epoch-14.pth \
    --checkpoint-dir checkpoint/stage2 \
    --log-dir log/stage2 \
    --epoch-start 15 \
    --epoch-end 17
```

### Stage 3
```sh
python train.py \
    --model-variant unireplknet \
    --dataset videomatte \
    --train-hr \
    --resolution-lr 512 \
    --resolution-hr 2048 \
    --seq-length-lr 20 \
    --seq-length-hr 3 \
    --learning-rate-backbone 0.00001 \
    --learning-rate-aspp 0.00001 \
    --learning-rate-decoder 0.00001 \
    --learning-rate-refiner 0.0002 \
    --checkpoint checkpoint/stage2/epoch-16.pth \
    --checkpoint-dir checkpoint/stage3 \
    --log-dir log/stage3 \
    --epoch-start 17 \
    --epoch-end 19
```

### Stage 4
```sh
python train.py \
    --model-variant unireplknet \
    --dataset imagematte \
    --train-hr \
    --resolution-lr 512 \
    --resolution-hr 2048 \
    --seq-length-lr 20 \
    --seq-length-hr 3 \
    --learning-rate-backbone 0.00001 \
    --learning-rate-aspp 0.00001 \
    --learning-rate-decoder 0.00005 \
    --learning-rate-refiner 0.0002 \
    --checkpoint checkpoint/stage3/epoch-18.pth \
    --checkpoint-dir checkpoint/stage4 \
    --log-dir log/stage4 \
    --epoch-start 19 \
    --epoch-end 24
```

<br><br><br>

## Evaluation

* [videomatte_512x512.tar (PNG 1.8G)](https://robustvideomatting.blob.core.windows.net/eval/videomatte_512x288.tar)
* [videomatte_1920x1080.tar (JPG 2.2G)](https://robustvideomatting.blob.core.windows.net/eval/videomatte_1920x1080.tar)
* [videomatt](https://drive.google.com/file/d/1QT4KHeGW3YrtBs1_7zovdCwCAofQ_GIj/viewusp=sharing)
* [CRGNN-R](https://www.dropbox.com/sh/23uvsue5we7e7b5/AAB4GSSWIaKiSouvN3wuWiwWa?dl=0)

Evaluation scripts are provided in `/evaluation` folder.