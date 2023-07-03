import sys
sys.path.append("..")
from models import *
import base64


def name2net(name, n_channels, n_classes):
    net_dict = {
            # '模型名称'：(模型结构， 模型对输入图片分辨率的限制H, 模型对输入图片分辨率的限制W)
            "UNet": (UNet(n_channels=n_channels, n_classes=n_classes), None, None)
    }
    return None if name not in net_dict else net_dict[name]


def video2base64(path):
    with open(path, mode="rb") as f:
        bytes_buffer = f.read()
        return base64.standard_b64encode(bytes_buffer)
