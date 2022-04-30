# code in this file is adapted from
# https://github.com/ildoonet/pytorch-randaugment/blob/master/RandAugment/augmentations.py
# https://github.com/google-research/fixmatch/blob/master/third_party/auto_augment/augmentations.py

# backup https://github.com/AhmadQasim/FixMatch/blob/master/dataset/randaugment.py 0.5 probability do augmentation

import random
import numpy as np
import PIL
import PIL.ImageOps
import PIL.ImageEnhance
import PIL.ImageDraw
from PIL import Image
import math
import torch
from torchvision import transforms

PARAMETER_MAX = 10

def Autocontrast(img, param):
    enh = PIL.ImageOps.autocontrast(img.convert("RGB"))
    return enh

def Brightness(img, param):
    enh = PIL.ImageEnhance.Brightness(img).enhance(param)
    return enh

def Color(img, param):
    enh = PIL.ImageEnhance.Color(img).enhance(param)
    return enh

def Contrast(img, param):
    enh = PIL.ImageEnhance.Contrast(img).enhance(param)
    return enh

def Equalize(img, param):
    enh = PIL.ImageOps.equalize(img.convert("RGB"))
    return enh

def Identity(img, param):
    return img

def Posterize(img, param):
    enh = PIL.ImageOps.posterize(img.convert("RGB"), param)
    return enh

def Rotate(img, param):
    if random.random() >0.5:
        param = -param
    enh = img.rotate(param)
    return enh

def Sharpness(img, param):
    enh = PIL.ImageEnhance.Sharpness(img).enhance(param)
    return enh

def Shear_x(img,param):
    if random.random() >0.5:
        param = -param
    enh = img.transform(img.size, PIL.Image.AFFINE, (1, param, 0, 0, 1, 0))
    return enh

def Shear_y(img,param):
    if random.random() >0.5:
        param = -param
    enh = img.transform(img.size, PIL.Image.AFFINE, (1, 0, 0, param, 1, 0))
    return enh

def Solarize(img, param):
    enh = PIL.ImageOps.solarize(img.convert("RGB"), 256 - param)
    return enh

def Translate_x(img,param):
    if random.random() >0.5:
        param = -param
    width = img.size[0]
    v = param *width
    enh = img.transform(img.size, PIL.Image.AFFINE, (1, 0, v, 0, 1, 0))
    return enh

def Translate_y(img,param):
    if random.random() >0.5:
        param = -param
    height = img.size[1]
    v = param * height
    enh = img.transform(img.size, PIL.Image.AFFINE, (1, 0, 0, 0, 1, v))
    return enh

def CutoutAbs(img, v):
    w, h = img.size
    x0 = np.random.uniform(0, w)
    y0 = np.random.uniform(0, h)
    x0 = int(max(0, x0 - v / 2.))
    y0 = int(max(0, y0 - v / 2.))
    x1 = int(min(w, x0 + v))
    y1 = int(min(h, y0 + v))
    xy = (x0, y0, x1, y1)
    # gray
    color = (127, 127, 127)
    img = img.copy()
    PIL.ImageDraw.Draw(img).rectangle(xy, color)
    return img

def Grid(d1, d2, rotate, ratio, prob, img):
    h, w = img.size
    img = transforms.ToTensor()(np.array(img)) 
    if np.random.rand() > prob:
        return img
    #Chengming added
    hh = math.ceil((math.sqrt(h*h + w*w)))

    d = np.random.randint(d1, d2)
    
    l = math.ceil(d*ratio)
    
    mask = np.ones((hh, hh), np.float32)
    st_h = np.random.randint(d)
    st_w = np.random.randint(d)
    for i in range(-1, hh//d+1):
            s = d*i + st_h
            t = s+l
            s = max(min(s, hh), 0)
            t = max(min(t, hh), 0)
            mask[s:t,:] *= 0
    for i in range(-1, hh//d+1):
            s = d*i + st_w
            t = s+l
            s = max(min(s, hh), 0)
            t = max(min(t, hh), 0)
            mask[:,s:t] *= 0
    r = np.random.randint(rotate)
    mask = Image.fromarray(np.uint8(mask))
    mask = mask.rotate(r)
    mask = np.asarray(mask)
    mask = mask[(hh-h)//2:(hh-h)//2+h, (hh-w)//2:(hh-w)//2+w]

    mask = torch.from_numpy(np.array(mask)).float()
    mask = mask.expand_as(img)
    img = img * (1-mask) 
    
    return img


def GridMask(d1, d2, rotate, ratio, prob, img):
    #grid = Grid(d1, d2, rotate, ratio, prob, img)
    #h,w = img.size
    #c = 3
    #y = []
    #y.append(Grid(d1, d2, rotate, ratio, prob, img))
    #y = torch.cat(y).view(c,h,w)
    y = Grid(d1, d2, rotate, ratio, prob, img)
    #print(type(y))
    y = transforms.ToPILImage()(y)
    return y

def augmentList():
    augs = [
            (Autocontrast,  None, None,  None),
            (Brightness,    0.05, 0.9,  float),
            (Color,         0.05, 0.9,  float),
            (Contrast,      0.05, 0.9,  float),
            (Equalize,      None, None,  None),
            (Identity,      None, None,  None),
            (Posterize,         4, 8,     int),
            (Rotate,        0,    30,     int),  #range:[-30,30]
            (Sharpness,     0.05, 0.9,  float),
            (Shear_x,       0,    0.3,  float),  #range:[-0.3, 0.3]
            (Shear_y,       0,    0.3,  float),  #range:[-0.3, 0.3]
            (Solarize,      0,    256,    int),
            (Translate_x,   0,    0.3,  float),  #range:[-0.3, 0.3]
            (Translate_y,   0,    0.3,  float)   #range:[-0.3, 0.3]
    ]
    return augs

def int_param(level, range_min, range_max):
    inter_size = range_max - range_min
    param = int(level * inter_size / PARAMETER_MAX) + range_min
    return param

def float_param(level, range_min, range_max):
    inter_size = range_max - range_min
    param = float(level) * inter_size / PARAMETER_MAX + range_min
    return param

class RandAugment(object):
    """Generate a set of distortions.
    Args:
        N: Number of augmentation
        M: Magnitude
    """
    def __init__(self, n, m, cut_type):
        self.n = n
        self.m = m
        self.cut_type = cut_type
        self.augmentList = augmentList()

    def __call__(self, img):
        operations = random.choices(self.augmentList, k=self.n)
        for op_name, range_min,range_max, flag in operations:
            param = None
            level = np.random.randint(0, self.m)
            if flag == int:
                param = int_param(level, range_min, range_max)
                assert range_min <= param <= range_max
            elif flag == float:
                param = float_param(level, range_min, range_max)
                assert range_min <= param <= range_max
            img = op_name(img, param)
            # print("{} {}".format(op_name, param))
        if self.cut_type == "cutout":
            img = CutoutAbs(img, 16)
        elif self.cut_type == "gridmask":
            img = GridMask(16, 32, 1, 0.4, 0.5, img)
        return img


if __name__ == '__main__':
    img = Image.open('/Users/cil/Documents/DL_advanced/others/DD2412Project/datasets/test.png')
    ra = RandAugment(2,10,cut_type)
    t = ra(img)
    t.show()
    # t = Contrast(img, 0.05)
    # t.show()
    # t2 = AutoContrast(img)
    # t2.show()
    # t = Brightness(img, 1)
    # t.show()
    # t = Color(img, 1)
    # t.show()
    # t = Equalize(img)
    # t.show()
    # t = Posterize(img,8)
    # t.show()
    # t = Rotate(img,30)
    # t.show()
    # t = Sharpness(img,1)
    # t.show()
    # t = Shear_x(img,0.1)
    # t.show()
    # t = Shear_y(img, 0.1)
    # t.show()
    # t = Solarize(img, 128)
    # t.show()
    # t = Translate_x(img,0.5)
    # t.show()
    # t = Translate_y(img, 0.5)
    # t.show()
