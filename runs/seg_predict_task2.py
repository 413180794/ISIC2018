# 该代码模型与实现参考了  https://github.com/chvlyl/ISIC2018
# 具体实现请移步至上述仓库

# 主要实现了ISIC-2018中的任务2，Lession attribute detection。
# 本代码功能只在于实现功能，对训练，模型生成并不涉及。
import os

import cv2
import torch
import torch.nn.functional as F
from PIL import Image
from skimage import io
from torch.backends import cudnn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import numpy as np
from models.unet16 import UNet16
from paths import model_data_dir


class TestDataset(Dataset):
    def __init__(self, image_paths, transform=None):
        self.image_paths = image_paths
        self.transform = transform
        self.n = len(image_paths)

    def __len__(self):
        """
        This function gets called with len()

        1. The length should be a deterministic function of some instance variables and should be a non-ambiguous representation of the total sample count. This gets tricky especially when certain samples are randomly generated, be careful
        2. This method should be O(1) and contain no heavy-lifting. Ideally, just return a pre-computed variable during the constructor call.
        3. Make sure to override this method in further derived classes to avoid unexpected samplings.
        """
        return self.n

    def __getitem__(self, index):
        ### load image
        image_file = self.image_paths[index]
        img_np, W, H = load_image_from_file(image_file)
        return img_np, W, H


def get_list_filename_extension(file_path):
    # 返回文件目录，文件名，文件扩展名
    (filepath, tempfilename) = os.path.split(file_path)
    (filename, extension) = os.path.splitext(tempfilename)
    return filepath, filename, extension


def load_image_from_file(image_path):
    # 读取图片文件，返回成可使用格式
    img_np = io.imread(image_path)
    # img = img.convert('RGB')
    img_np = np.asarray(img_np, dtype=np.float)
    img_np = (img_np / 255).astype('float32')
    if len(img_np.shape) == 2:
        img_np = img_np[:, :, np.newaxis]
    (H, W, C) = img_np.shape
    # print(img_np.shape)
    img_np = cv2.resize(img_np, (512, 512), interpolation=cv2.INTER_CUBIC)
    # 返回图片的二维数组、宽、高
    return img_np, W, H

def seg_predict_image_task2(image_paths):
    '''
    输入一张图片的路径，输出预测的结果
    :param image_path:
    :return:
    '''
    data = TestDataset(image_paths)
    test_data = DataLoader(data, batch_size=1, shuffle=False, num_workers=10, pin_memory=False)
    model = UNet16(num_classes=5, pretrained='vgg')
    device = 'cpu'  # 使用cpu
    model.to(device)
    model_weight = os.path.join(model_data_dir, 'task2_vgg16_k0_v0', 'task2_vgg16_k0_v0.ckpt')  # 模型权重的地址
    state = torch.load(model_weight, map_location='cpu')  # model_weight为训练出的模型权重
    new_state = {}

    for k, v in state['model'].items():
        if str(k).startswith("module."):
            k = k[7:]
        new_state[k] = v

    state['model'] = new_state
    model.load_state_dict(state['model'])
    cudnn.benchmark = True
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])
    attr_types = ['pigment_network', 'negative_network', 'streaks', 'milia_like_cyst', 'globules']
    alpha = 0.5
    origin_image_np = np.asarray(Image.open(image_paths[0]))
    with torch.no_grad():
        for test_image, W, H in test_data:
            # origin_image_np = test_image
            test_image = test_image.to(device)

            test_image = test_image.permute(0, 3, 1, 2)
            outputs, outputs_mask_ind1, outputs_mask_ind2 = model(test_image)
            test_prob = F.sigmoid(outputs)
            test_prob = test_prob.squeeze().data.cpu().numpy()
            for ind, attr in enumerate(attr_types):
                resize_mask = cv2.resize(test_prob[ind, :, :], (W, H), interpolation=cv2.INTER_CUBIC)
                # for cutoff in [0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9]:
                for cutoff in [0.3]:
                    test_mask = (resize_mask > cutoff).astype('int') * 255
                    origin_image_np = put_predict_image(origin_image_np,test_mask,attr,alpha)
            origin_image_np = cv2.resize(origin_image_np,(W,H),interpolation=cv2.INTER_CUBIC)
            print({
                "image_np":origin_image_np,
                "width":W,
                "height":H
            })

def save_picture(image_path,image_np):
    u
    b, g, r = cv2.split(image_np)
    cv2.imwrite(image_path,cv2.merge([r,g,b]))

attr_colors = {
    'pigment_network': (0, 107, 176),
    'negative_network': (239, 169, 13),
    'streaks': (29, 24, 21),
    'milia_like_cyst': (5, 147, 65),
    'globules': (220, 47, 31)
}


def put_predict_image(origin_image_np, test_mask, attr, alpha):
    '''
    将predict图片以apha透明度覆盖到origin图片中
    :param origin_image:
    :param predict_image:
    :param RGB:
    :param alpha:
    :return:
    '''

    test_mask_RGB = Image.fromarray(test_mask.astype('uint8')).convert("RGB") # 将原始二值化图像转换成RGB
    # origin_image_np = np.asarray(origin_image_np,dtype=np.uint8)
    test_mask_np = np.asarray(test_mask_RGB,dtype=np.uint8) # 将二值化图像转换成三维数组
    height, width, channels = test_mask_np.shape  # 获得图片的三个纬度
    # 转换预测图像的颜色
    origin_image_np.flags.writeable=True
    test_mask_np.flags.writeable = True
    for row in range(height):
        for col in range(width):
            # 上色
            if test_mask_np[row, col, 0] == 255 and test_mask_np[row, col, 1] == 255 and test_mask_np[row, col, 2] == 255:
                test_mask_np[row, col, 0] = attr_colors[attr][0]
                test_mask_np[row, col, 1] = attr_colors[attr][1]
                test_mask_np[row, col, 2] = attr_colors[attr][2]

            if test_mask_np[row, col, 0] != 0 or test_mask_np[row,col, 1] != 0 or test_mask_np[row, col, 2] != 0:
                origin_image_np[row,col,0] = alpha*origin_image_np[row,col,0] + (1-alpha)*test_mask_np[row, col, 0]
                origin_image_np[row,col,1] = alpha*origin_image_np[row,col,1] + (1-alpha)*test_mask_np[row, col, 1]
                origin_image_np[row,col,2] = alpha*origin_image_np[row,col,2] + (1-alpha)*test_mask_np[row, col, 2]
    return origin_image_np
if __name__ == '__main__':
    seg_predict_image_task2(["/home/zhangfan/workData/LinuxCode/pythonProject/ISIC2018/datasets/ISIC2018/data/ISIC2018_Task1-2_Training_Input/ISIC_0000031.jpg",])
