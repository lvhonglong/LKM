# **Real-time Video Portrait Matting Method Based on Large-Kernel Convolutions**

![Teaser](/documentation/image/teaser.gif)

<p align="center">English | <a href="README_zh_Hans.md">中文</a></p>

The GitHub repository for the paper **Real-time Video Portrait Matting Method Based on Large-Kernel Convolutions**. LKM is designed for high-precision, real-time video matting of portraits. This paper proposes a video portrait matting network based on large-kernel convolution, achieving high-precision real-time video portrait matting through the use of large-kernel convolutions, wavelet transforms, and re-parameterization techniques. The network employs large-kernel convolutions to enhance the encoding and decoding capabilities of image features. Additionally, the wavelet transform is used to pass image information to the decoding network, providing more useful information. Finally, re-parameterization technology is employed to fuse multiple convolution branches with large kernels in parallel, effectively reducing computational burden and speeding up inference. The model can achieve processing speeds of 103 fps for HD video and 94 fps for 4K video on an Nvidia GTX 3090 GPU, enabling real-time video matting.
<br>


## Environment Setup

1. Unzip the package and create a virtual environment:
```sh
unzip LKM-main.zip
cd LKM-main
conda create -n LKM python=3.7
conda activate LKM
```

2. Install the corresponding version of PyTorch for CUDA. Here we are using version CUDA 11.1:

```sh
pip install torch==1.10.1+cu111 torchvision==0.11.2+cu111 torchaudio==0.10.1 -f https://download.pytorch.org/whl/cu111/torch_stable.html
```

3. Install the necessary library files for using unireplknet:

```sh
pip install -U openmim
mim install mmcv-full==1.5.0
mim install mmsegmentation==0.27.0
pip install timm==0.6.11 mmdet==2.28.1
```

4. Install other library files:
```sh
pip install -r requirements.txt
```

<br>

## Training and Evaluation

Please refer to the [training documentation](documentation/training.md) to train and evaluate your own model.

<br>

## Demo

video:

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
picture:

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


wecab:

```python
python inference_wecab.py
```

<br>

## Speed

Speed is measured with `inference_speed_test.py` for reference.

| GPU            | dType | HD (1920x1080) | 4K (3840x2160) |
| -------------- | ----- | -------------- |----------------|
| RTX 3090       | FP16  | 103 FPS        | 94 FPS        |


* Note 1: HD uses `downsample_ratio=0.25`, 4K uses `downsample_ratio=0.125`. All tests use batch size 1 and frame chunk 1.
* Note 2: We refer to the method used by RVM for testing speed by measuring only tensor throughput. Directly using a camera for testing may reduce the speed, but it still maintains a fast processing capability, achieving real-time inference. The use of video hardware encoding and decoding technology could further improve the inference speed. For details, please refer to [PyNvCodec](https://github.com/NVIDIA/VideoProcessingFramework).

<br>  


<br>

