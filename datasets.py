# 数据集文件，需要分别对两种风格的图片进行读取
import glob
from PIL import Image
from torch.utils.data import Dataset
import torchvision.transforms as transforms


class ImageDataset(Dataset):
    def __init__(self, root, transforms_=None):
        self.transform = transforms.Compose(transforms_)                                  # 串联多张图片变换的操作，Compose()类会将transforms列表里面的transform操作进行遍历
        self.files = sorted(glob.glob(root + "*.*"))                                      # glob()：可以将指定目录下所有跟通配符模式相同的文件放到一个列表中；sorted()：排序函数

    def __getitem__(self, index):
        img = Image.open(self.files[index % len(self.files)])                             # 加载训练集
        w, h = img.size
        img_A = img.crop((0, 0, w/3, h))                                                  # Image.crop(left, up, right, below)
        img_B = img.crop((w/3, 0, 2*w/3, h))
        img_C = img.crop((2*w/3, 0, w, h))
        img_A = self.transform(img_A)                                                     # 对第二层解密图像进行变换
        img_B = self.transform(img_B)                                                     # 对第一层解密图像进行变换
        img_C = self.transform(img_C)                                                     # 对第一层解密图像进行变换
        return {"A": img_A, "B": img_B, "C": img_C}

    def __len__(self):
        return len(self.files)
