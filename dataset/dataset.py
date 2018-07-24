import torch
from torch.utils import data
import torchvision.transforms as transforms

import numpy as np
import os.path as osp
from PIL import Image
from PIL import ImageFile
import joint_transforms
ImageFile.LOAD_TRUNCATED_IMAGES = True


class PRCVData(data.Dataset):
    def __init__(self, mode, root, label_list, trans=None):
        self.mode = mode
        self.root = root
        self.label_list = [i.strip() for i in open(label_list)]
        self.transforms = trans
        self.im2te = ImToTensor()
        self.lb2te = MaskToTensor()

    def __len__(self):
        return len(self.label_list)

    def __getitem__(self, item):
        img_name = osp.join(self.root, self.label_list[item].split(' ')[0])
        label_name = osp.join(self.root, self.label_list[item].split(' ')[-1])
        im, gt = Image.open(img_name).convert('RGB'), Image.open(label_name)
        if self.transforms is not None:
            im, gt = self.transforms(im, gt)
        return self.im2te(im), self.lb2te(gt)


class ImToTensor(object):
    def __call__(self, im):
        trans = transforms.Compose([
            transforms.ToTensor(),
            # transforms.Normalize(mean=[x/255.0 for x in [85.86, 91.79, 85.00]],
            #                      std=[x/255.0 for x in [35.79, 35.13, 36.51]]),
        ])
        return trans(im)


class MaskToTensor(object):
    def __call__(self, gt):
        return torch.from_numpy(np.array(gt, dtype=np.int32)).long()


if __name__ == '__main__':
    transform_train = joint_transforms.Compose([
            joint_transforms.Scale(512),
    ])
    dataset = PRCVData('train', '/data/jinqizhao/', './list/tank_val_list.txt', trans=transform_train)
    for i in range(len(dataset)):
        im, gt = dataset[i]
        print(i)
    print(1)
