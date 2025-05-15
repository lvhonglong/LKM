# **Real-time Video Portrait Matting Method Based on Large-Kernel Convolutions**

![Teaser](/documentation/image/teaser.gif)

<p align="center"><a href="README.md">English</a> | 中文</p>

论文 **Real-time Video Portrait Matting Method Based on Large-Kernel Convolutions**的 GitHub 库。LKM 为高精度实时人像视频抠像设计。本文提出了一种基于大核卷积设计的视频人像抠图网络，通过使用大核卷积、小波变换和重参数化等技术，实现了端到端的高精度实时视频人像抠图。网络中使用了大核卷积，以提升对图像特征的编解码能力。同时，通过小波变换将图像的信息传递给解码网络，提供更多的有用信息。最后，采用重参数化技术将大卷积核并行的多个卷积分支进行融合，有效减少了计算负担，加快了推理速度。模型可以在Nvidia GTX 3090 GPU上实现**HD 103 fps**和**4K 94 fps**的处理速度，实现实时视频抠图。

<br>

## 更新

* 

<br>

<br>

## 环境配置

1. 解压包并创建虚拟环境:
```sh
unzip LKM-main.zip
cd LKM-main
conda create -n LKM python=3.7
conda activate LKM
```

2. 安装相应cuda版本的Pytorch，这里使用了cuda11.1版本:

```sh
pip install torch==1.10.1+cu111 torchvision==0.11.2+cu111 torchaudio==0.10.1 -f https://download.pytorch.org/whl/cu111/torch_stable.html
```

3. 安装使用unireplknet需要的库文件：

```sh
pip install -U openmim
mim install mmcv-full==1.5.0
mim install mmsegmentation==0.27.0
pip install timm==0.6.11 mmdet==2.28.1
```

4. 安装其他库文件:
```sh
pip install -r requirements.txt
```

<br>

## 训练和评估

请参照[训练文档（英文）](documentation/training.md)。

<br>

## Demo

视频:

```python
python inference.py \
  --variant unireplknet \
  --checkpoint "./pretrained/rep_model.pth" \
  --device cuda \
  --input-source "input.mp4" \
  --output-type video \
  --output-composition "out_com.mp4" \
  --output-alpha "out_pha.mp4" \
  --output-foreground "out_fgr.mp4" \
  --output-video-mbps 4 \
  --seq-chunk 12
```
图片:

```python
python inference.py \
    --variant mobilenetv3 \
    --checkpoint "./pretrained/rep_model.pth" \
    --device cuda \
    --input-source ./your_folder \
    --output-type png_sequence \
    --output-composition ./your_folder_com \
    --output-alpha ./your_folder_alpha \
    --output-foreground ./your_folder_fgr \
    --output-video-mbps 4 \
    --seq-chunk 1
```

摄像头：

```python
python inference_wecab.py
```

<br>

## 速度

速度用 `inference_speed_test.py` 测量以供参考。

| GPU            | dType | HD (1920x1080) | 4K (3840x2160) |
| -------------- | ----- | -------------- |----------------|
| RTX 3090       | FP16  | 103 FPS        | 94 FPS        |

* 注释1：HD 使用 `downsample_ratio=0.25`，4K 使用 `downsample_ratio=0.125`。 所有测试都使用 batch size 1 和 frame chunk 1。
* 注释2：我们参照rvm测试速度的方法，只测量张量吞吐量（tensor throughput）。 直接使用摄像头进行测试速度会降低，但仍能保持较快的处理速度，实现实时推理。使用视频硬件编解码技术可能会进一步改善推理速度，详细请参考 [PyNvCodec](https://github.com/NVIDIA/VideoProcessingFramework)。

<br>



<br>



