import torch
from model.model_upsample import MattingNetwork

from inference import convert_video

model = MattingNetwork('unireplknet',deploy= True).eval().cuda()  # rep_model
model.load_state_dict(torch.load('./pretrained/rep_model.pth'), strict=False)

convert_video(
    model,                           # The model, can be on any device (cpu or cuda).
    input_source='input.mp4',        # A video file or an image sequence directory.
    output_type='video',             # Choose "video" or "png_sequence"
    output_composition='com.mp4',    # File path if video; directory path if png sequence.
    output_alpha="pha.mp4",          # [Optional] Output the raw alpha prediction.
    output_foreground="fgr.mp4",     # [Optional] Output the raw foreground prediction.
    output_video_mbps=4,             # Output video mbps. Not needed for png sequence.
    downsample_ratio=None,           # A hyperparameter to adjust or use None for auto.
    seq_chunk=12,                    # Process n frames at once for better parallelism.
)
