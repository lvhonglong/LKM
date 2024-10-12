import torch
from model.model_upsample import MattingNetwork
from inference import convert_video
import os

# model = MattingNetwork('unireplknet').eval().cuda()  # or "resnet50"
# model.load_state_dict(torch.load('./pretrained/model.pth'))

model = MattingNetwork('unireplknet',deploy= True).eval().cuda()  # rep_model
model.load_state_dict(torch.load('./pretrained/rep_model.pth'), strict=False)





def convert_video_batch(
    model,
    input_base='./videomatte_512x288/videomatte_static/',
    output_type='png_sequence',
    output_base='/videomatte_512x288_out/videomatte_static/',
    downsample_ratio=None,
    seq_chunk=12,
):
    for i in range(0, 25):  # Assuming you want to process files from 0005 to 0024
        input_source = os.path.join(input_base, f"{i:04d}", "com")
        output_composition = os.path.join(output_base, f"{i:04d}", "com")
        output_alpha = os.path.join(output_base, f"{i:04d}", "pha")
        output_foreground = os.path.join(output_base, f"{i:04d}", "fgr")

        convert_video(
            model=model,
            input_source=input_source,
            output_type=output_type,
            output_composition=output_composition,
            output_alpha=output_alpha,
            output_foreground=output_foreground,
            downsample_ratio=downsample_ratio,
            seq_chunk=seq_chunk,
        )

# 调用这个函数来处理所有视频
convert_video_batch(
    model=model,
    input_base='./videomatte_512x288/videomatte_motion/',
    output_type='png_sequence',
    output_base='./videomatte_512x288_out/videomatte_static/',
    downsample_ratio=None,
    seq_chunk=12,
)