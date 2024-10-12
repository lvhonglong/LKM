import torch
import torch.nn as nn

from model_upsample import MattingNetwork

# 创建MattingNetwork实例
matting_net = MattingNetwork('unireplknet')

# 遍历模块
for module in matting_net.modules():
    # 打印模块的名称和类型
    print(module.__class__.__name__)
# 加载已训练好的模型
trained_model = MattingNetwork('unireplknet')  # 创建 UniRepLKNet 模型的实例
trained_model.load_state_dict(torch.load('./pretrained/model.pth'))  # 加载已训练好的权重

for trained_module in trained_model.modules():
    if hasattr(trained_module, 'reparameterize'):
                trained_module.reparameterize()
torch.save(trained_model.state_dict(), './pretrained/rep_model.pth')
