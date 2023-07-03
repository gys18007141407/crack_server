from os.path import splitext
from os import listdir
import numpy as np
from glob import glob
import torch
from torch.utils.data import Dataset
import logging
from PIL import Image


class CrackDataset(Dataset):
    def __init__(self, imgs_dir, labels_dir):
        self.imgs_dir = imgs_dir
        self.labels_dir = labels_dir
        self.ids = [splitext(file)[0] for file in listdir(imgs_dir)
                    if not file.startswith('.')]
        logging.info(f'there are {len(self.ids)} pictures in datasets')

    def __len__(self):
        return len(self.ids)

    @classmethod
    def preprocess(cls, pil_img):
        img_nd = np.array(pil_img)
        
        # 单通道图像
        if len(img_nd.shape) == 2:
            img_nd = np.expand_dims(img_nd, axis=2)

        # [H, W, C] ==> [C, H, W]
        img_trans = img_nd.transpose((2, 0, 1))
        img_trans = img_trans / 255
        return img_trans

    def __getitem__(self, i):
        img_prefix_name = self.ids[i]
        # glob获取满足表达式的路径集合
        label_file = glob(self.labels_dir + img_prefix_name + '.*')
        img_file = glob(self.imgs_dir + img_prefix_name + '.*')

        # img和label一一对应
        assert len(label_file) == 1, \
            f'Either no label or multiple labels found for the ID {img_prefix_name}: {label_file}'
        assert len(img_file) == 1, \
            f'Either no image or multiple images found for the ID {img_prefix_name}: {img_file}'

        # 读取图片
        label = Image.open(label_file[0])
        img = Image.open(img_file[0])

        assert img.size == label.size, \
            f'Image and label {img_prefix_name} should be the same size, but are {img.size} and {label.size}'

        # 预处理
        img = self.preprocess(img)
        label = self.preprocess(label)

        return {
            'image': torch.from_numpy(img).type(torch.FloatTensor),
            'label': torch.from_numpy(label).type(torch.FloatTensor)
        }

