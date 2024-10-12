from .Unireplknet import UniRepLKNet
import torch
from torch import nn
from torchvision.transforms.functional import normalize

class UniRepLKNetEncoder(nn.Module):
    def __init__(self, pretrained_backbone=False, deploy = False):
        super().__init__()
        if pretrained_backbone:
            init_cfg ={ 'checkpoint': './pretrained/unireplknet_a.pth' } # 替换为有效的字符串路径}
            self.unirep = UniRepLKNet(init_cfg =init_cfg, deploy = deploy)
        else:
            self.unirep = UniRepLKNet(deploy = deploy)  # 使用UniRepLKNet特征提取网络

    def forward_single_frame(self, x):
        x = normalize(x, [0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        features = self.unirep(x)
        f1 = features[0]
        f2 = features[1]
        f3 = features[2]
        f4 = features[3]
        return [f1, f2, f3, f4]

    def forward_time_series(self, x):
        B, T = x.shape[:2]
        features = self.forward_single_frame(x.flatten(0, 1))
        features = [f.unflatten(0, (B, T)) for f in features]

        return features

    def forward(self, x):
        if x.ndim == 5:
            return self.forward_time_series(x)
        else:
            return self.forward_single_frame(x)
