# -*- coding: utf-8 -*-
import json
# 调用python安装的thrift依赖包
import os

import cv2
import torch
from PIL import Image
import tensorflow as tf
from skimage import transform
import numpy as np
import thriftpy2
from thriftpy2.protocol import TBinaryProtocolFactory
from thriftpy2.transport import TBufferedTransportFactory
from torch.backends import cudnn
from torch.utils.data import DataLoader, Dataset
import torch.nn.functional as F
from misc_utils.prediction_utils import inv_sigmoid, sigmoid
from models import backbone
from models.unet16 import UNet16
from paths import task2_model_name, task1_model_name, task3_model_name, task1_result_dir, task2_result_dir
from runs.seg_eval import task1_post_process
from keras import Model
predict_thrift = thriftpy2.load("server.thrift", module_name="predict_thrift")
from thriftpy2.rpc import make_server


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

    test_mask_np = np.asarray(test_mask_RGB,dtype=np.int) # 将二值化图像转换成三维数组
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

def get_fileDirectoryPath_fileName_fileExt(filePath):
    """
    获取文件目录路径， 文件名， 后缀名
    :param fileUrl:
    :return:
    """
    fileDirectoryPath, tmpfilename = os.path.split(filePath)
    fileName, extension = os.path.splitext(tmpfilename)
    return fileDirectoryPath, fileName, extension

def load_image_from_file(image_file):
    img = Image.open(image_file)
    img = img.convert('RGB')
    img_np = np.asarray(img, dtype=np.float)
    ### why only 0-255 integers
    img_np = (img_np / 255.0).astype('float32')
    ### resize the image
    #img_np = cv2.resize(img_np, (512, 512), interpolation=cv2.INTER_CUBIC)
    if len(img_np.shape) == 2:
        img_np = img_np[:, :, np.newaxis]
    (H, W, C) = img_np.shape
    # print(img_np.shape)
    img_np = cv2.resize(img_np, (512, 512), interpolation=cv2.INTER_CUBIC)

    return img_np, W, H
def get_resize_image_np(image_path, output_size=512):
    # 得到改变形状的图片
    image = Image.open(image_path)
    if image.mode != "RGB":
        image = image.convert('RGB')
    # 转换为np
    image_np = (np.asarray(image,dtype=np.float)/255.0).astype('float32')
    # 得到图片宽、高、通道书
    W, H, channel = image_np.shape
    # 将图片大小转换为(output_size,output_size)
    resize_image_np = transform.resize(image_np, (output_size, output_size),
                                       order=1, mode='constant',
                                       cval=0, clip=True,
                                       preserve_range=True)

    return resize_image_np.astype(np.uint8), W, H

def get_resize_images_np(image_paths,output_size=224):
    resize_images_np =  []
    images_size = []
    for image_path in image_paths:
        image = Image.open(image_path)
        if image.mode != "RGB":
            image = image.convert("RGB")
        image_np = np.asarray(image)
        W,H,_ = image_np.shape
        resize_image_np = transform.resize(image_np,(output_size,output_size),
                                           order=1,mode='constant',
                                           cval=0,clip=True,
                                           preserve_range=True)
        resize_images_np.append(resize_image_np)
        images_size.append((W,H))
    resize_images_np = np.stack(resize_images_np).astype(np.uint8)
    return resize_images_np,images_size

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
        _,image_name,image_ext = get_fileDirectoryPath_fileName_fileExt(image_file)
        img_np, W, H = load_image_from_file(image_file)
        return img_np, W, H,image_name,image_ext,image_file


def process_task2_model(model):
    model.to('cpu')
    model_state = torch.load(task2_model_name,map_location='cpu')
    new_state = {}
    for k,v in model_state['model'].items():
        if str(k).startswith("module."):
            k = k[7:]
        new_state[k] = v
    model_state['model'] = new_state
    model.load_state_dict(model_state['model'])
    cudnn.benchmark = True

def save_picture(image_path,image_np):
    b, g, r = cv2.split(image_np)
    cv2.imwrite(image_path,cv2.merge([r,g,b]))
class Handler:
    # 预先载入三个模型
    def __init__(self):
        self.graph = tf.get_default_graph()
        self.task1_model = backbone('vgg16').segmentation_model(load_from=task1_model_name)

        self.task2_model = UNet16(num_classes=5, pretrained='vgg')
        process_task2_model(self.task2_model)
        task3_model_ = backbone('inception_v3').classification_model(load_from=task3_model_name)
        self.task3_model = Model(inputs=task3_model_.input,outputs=task3_model_.get_layer('predictions').output)


    def seg_predict_images_task1(self,image_paths):
        results = []
        resize_images_np,images_size = get_resize_images_np(image_paths)
        images_num = len(images_size)
        y_pred = np.zeros(shape=(images_num,224,224))
        with self.graph.as_default():
            y_pred += inv_sigmoid(self.task1_model.predict_on_batch(resize_images_np))[:, :, :, 0]
        y_pred = sigmoid(y_pred)
        y_pred = task1_post_process(y_prediction=y_pred,threshold=0.5,gauss_sigma=2.)
        for index,image_path in enumerate(image_paths):
            image_dir,image_name,image_ext = get_fileDirectoryPath_fileName_fileExt(image_path)
            current_pred = y_pred[index]
            current_pred = current_pred * 255
            resized_pred = transform.resize(current_pred,
                                            output_shape=images_size[index],
                                            preserve_range=True,
                                            mode='reflect')
            resized_pred[resized_pred > 128] = 255
            resized_pred[resized_pred <= 128] = 0
            img = Image.fromarray(resized_pred.astype(np.uint8))
            image_path = os.path.join(task1_result_dir,image_name+"_result"+image_ext)
            img.save(image_path)
            results.append(image_path)
        return results

    def seg_predict_images_task2(self,image_paths):
        test_datas = DataLoader(TestDataset(image_paths),batch_size=1,shuffle=False,num_workers=10,pin_memory=False)
        attr_types = ['pigment_network', 'negative_network', 'streaks', 'milia_like_cyst', 'globules']
        alpha = 0.5
        result = []
        with torch.no_grad():
            for test_image_np,W,H,image_name,image_ext,image_path in test_datas:
                origin_image_np = np.asarray(Image.open(image_path[0]))
                test_image_np = test_image_np.to('cpu').permute(0,3,1,2)
                outputs,outputs_mask_ind1,outputs_mask_ind2 = self.task2_model(test_image_np)
                test_prob = F.sigmoid(outputs).squeeze().data.cpu().numpy()
                for index,attr in enumerate(attr_types):
                    resize_mask = cv2.resize(test_prob[index, :, :], (W, H), interpolation=cv2.INTER_CUBIC)
                    for cutoff in [0.3]:
                        test_mask = (resize_mask > cutoff).astype('int') * 255
                        mask_image_path = os.path.join(task2_result_dir,image_name[0]+"_"+attr+image_ext[0])
                        result.append(mask_image_path)
                        cv2.imwrite(mask_image_path,test_mask)
                        # origin_image_np = put_predict_image(origin_image_np,test_mask,attr,alpha)
                result_image_path = os.path.join(task2_result_dir,image_name[0]+"_result"+image_ext[0])
                save_picture(result_image_path,origin_image_np)
                result.append(result_image_path)
        return result


    def cls_predict_images_task3(self,image_paths):

        predict_thrift.task3_result()
        results = []
        resize_images_np, images_size = get_resize_images_np(image_paths)
        images_num = len(images_size)
        y_pred = np.zeros(shape=(images_num, 7))
        with self.graph.as_default():
            y_pred += self.task3_model.predict_on_batch(resize_images_np)
        y_prob = sigmoid(y_pred)

        for index,image_path in enumerate(image_paths):
            task3_result = predict_thrift.task3_result()
            task3_result.name = image_path
            task3_result.result = y_prob[index]
            results.append(task3_result)
        return results

if __name__ == '__main__':
    server = make_server(predict_thrift.predictService,Handler(),
                         "127.0.0.1",8080,
                         proto_factory=TBinaryProtocolFactory(),
                         trans_factory=TBufferedTransportFactory(),client_timeout=None)
    print("start")
    server.serve()
    # server.trans.client_timeout = None
    # x = Handler()
    # print(x.seg_predict_images_task2([
    #     "/home/zhangfan/workData/LinuxCode/pythonProject/ISIC2018/datasets/ISIC2018/data/ISIC2018_Task1-2_Test_Input/ISIC_0012292.jpg", ]))
    # print(get_fileDirectoryPath_fileName_fileExt(
    #     "/home/zhangfan/workData/Linuxsode/pythonProject/ISIC2018/datasets/ISIC2018/data/ISIC2018_Task1-2_Test_Input/ISIC_0012292.ji"))
