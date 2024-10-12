import cv2
import time
from torchvision import transforms
from typing import Optional, Tuple
import torch
from model.model_upsample import MattingNetwork
 
 
 
def auto_downsample_ratio(h, w):
    """
    Automatically find a downsample ratio so that the largest side of the resolution be 512px.
    """
    return min(512 / max(h, w), 1)
 
 
def get_frame(num):
    cap = cv2.VideoCapture(num)
    while True:
        ret, frame = cap.read()
        fps= cap.get(cv2.CAP_PROP_FPS)
        print("摄像头帧速:", fps)
        yield frame
 
 
def convert_video(model,
                  input_resize: Optional[Tuple[int, int]] = None,
                  downsample_ratio: Optional[float] = None,
                  device: Optional[str] = None,
                  dtype: Optional[torch.dtype] = None):
    """
    Args:
        input_resize: If provided, the input are first resized to (w, h).
        downsample_ratio: The model's downsample_ratio hyperparameter. If not provided, model automatically set one.
        device: Only need to manually provide if model is a TorchScript freezed model.
        dtype: Only need to manually provide if model is a TorchScript freezed model.
    """
    assert downsample_ratio is None or (
                downsample_ratio > 0 and downsample_ratio <= 1), 'Downsample ratio must be between 0 (exclusive) and 1 (inclusive).'
 
 
    # Initialize transform
    if input_resize is not None:
        transform = transforms.Compose([
            transforms.Resize(input_resize[::-1]),
            transforms.ToTensor()
        ])
    else:
        transform = transforms.ToTensor()
 
    # Inference
    print("------------------------------------------------------------>")
    model = model.eval()
    if device is None or dtype is None:
        param = next(model.parameters())
        dtype = param.dtype
        device = param.device
 
    bgr = torch.tensor([120, 255, 155], device=device, dtype=dtype).div(255).view(1, 1, 3, 1, 1)
 
    with torch.no_grad():
        rec = [None] * 4
        for src in get_frame(-1):
            src = transform(src)
            src = src.unsqueeze(0)
 
            if downsample_ratio is None:
                downsample_ratio = auto_downsample_ratio(*src.shape[2:])
 
            src = src.to(device, dtype, non_blocking=True).unsqueeze(0)  # [B, T, C, H, W]
            t1 = time.time()
            fgr, pha, *rec = model(src, *rec, downsample_ratio)
            print("frame_cost:", (time.time() - t1) / src.shape[1])
            print("推理帧率：{:.2f}".format(1/((time.time() - t1) / src.shape[1])))
 
            com = fgr * pha + bgr * (1 - pha)
            frames = com[0]
            if frames.size(1) == 1:
                frames = frames.repeat(1, 3, 1, 1)  # convert grayscale to RGB
            frames = frames.mul(255).byte().cpu().permute(0, 2, 3, 1).numpy()  # [1, 480, 640, 3]
 
            yield frames[0]
 
 
def show_frame(frames):
    for frame in frames:
        cv2.imshow("capture", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            exit()
 
if __name__ == '__main__':
 
    # #-------测试摄像头是否可用------------#
    # for frame in get_frame(0):
    #     cv2.imshow("capture", frame)
    #     if cv2.waitKey(1) & 0xFF == ord('q'):
    #         break
    # #----------------------------------#
 
 
 
    model = MattingNetwork('unireplknet',deploy= True).eval().cuda()  # rep_model
    model.load_state_dict(torch.load('./pretrained/rep_model.pth'), strict=False)
 
    # 返回测试结果
    frames = convert_video(model)
 
    # 展示推理结果
    show_frame(frames)