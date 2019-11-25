import argparse

import numpy as np
from skimage import transform
from skimage import io
from misc_utils.prediction_utils import inv_sigmoid, cyclic_pooling, cyclic_stacking, sigmoid
from models import backbone


def task1_tta_predict(model, img_arr):
    img_arr_tta = cyclic_stacking(img_arr)
    mask_arr_tta = []
    for _img_crops in img_arr_tta:
        _mask_crops = model.predict(_img_crops)
        mask_arr_tta.append(_mask_crops)
    mask_crops_pred = cyclic_pooling(*mask_arr_tta)
    return mask_crops_pred


def get_resize_image_np(image_path, output_size):
    # 得到改变形状的图片
    image = io.imread(image_path)
    # 转换为np
    # image_np = np.asarray(image)
    # 得到图片宽、高、通道书
    W, H, channel = image.shape
    # 将图片大小转换为(output_size,output_size)
    resize_image = transform.resize(image, (output_size, output_size),
                                    order=1, mode='constant',
                                    cval=0, clip=True,
                                    preserve_range=True)
    resize_image_np = np.stack([resize_image, ]).astype(np.uint8)
    print(resize_image_np.shape)
    return resize_image_np, W, H


def seg_predict_images_task1(image_paths, use_tta=False):
    '''预测一组图片'''
    num_folds = 4
    use_tta = False
    image_num = len(image_paths)  # 得到图片的数量
    if use_tta:
        pass
    else:
        y_pred = np.zeros(shape=(1, 224, 224))
        run_name = 'task1_vgg16_k%d_v0' % num_folds
        model = backbone('vgg16').segmentation_model(load_from=run_name)
        for image_path in image_paths:
            resize_image_np, W, H = get_resize_image_np(image_path, 224)
            y_pred += inv_sigmoid(model.predict_on_batch(resize_image_np))[:, :, :, 0]
            y_pred = sigmoid(y_pred)
            current_pred = y_pred[0] * 255
            current_pred[current_pred > 128] = 255
            current_pred[current_pred <= 128] = 0
            print({
                "image_np": current_pred,
                "width": W,
                "height": H
            })


def seg_predict_image_task1(image_path, use_tta=False):
    '''只针对一张图片'''
    num_folds = 4
    # use_tta = False  # 如果为false,表示不采用取平均数方式
    y_pred = np.zeros(shape=(1, 224, 224))
    resize_image_np, W, H = get_resize_image_np(image_path, 224)
    if use_tta:
        # 采用取平均数的模式
        for k_fold in range(num_folds + 1):
            model_name = 'task1_vgg16'
            run_name = 'task1_vgg16_k%d_v0' % k_fold
            model = backbone('vgg16').segmentation_model(load_from=run_name)
            # model.load_weights("/home/zhangfan/workData/LinuxCode/pythonProject/ISIC2018/model_data/task1_vgg16_k0_v0/task1_vgg16_k0_v0.hdf5")
            y_pred += inv_sigmoid(model.predict_on_batch(resize_image_np))[:, :, :, 0]
        y_pred /= (num_folds + 1)
    else:
        # 只是用最后一组数据的权重
        run_name = 'task1_vgg16_k%d_v0' % num_folds
        model = backbone('vgg16').segmentation_model(load_from=run_name)
        y_pred = inv_sigmoid(model.predict_on_batch(resize_image_np))[:, :, :, 0]
    y_pred = sigmoid(y_pred)
    current_pred = y_pred[0] * 255
    current_pred[current_pred > 128] = 255
    current_pred[current_pred <= 128] = 0
    # 将resized_pred传给Nodejs
    print({
        "image_np": current_pred,
        "width": W,
        "height": H
    })
    # im = Image.fromarray(resized_pred.astype(np.uint8))


if __name__ == '__main__':
    seg_predict_image_task1("/home/zhangfan/workData/LinuxCode/pythonProject/ISIC2018/datasets/ISIC2018/data/ISIC2018_Task1-2_Test_Input/ISIC_0012236.jpg")
    # parser = argparse.ArgumentParser()
    # arg = parser.add_argument
    # arg('--image-path', type=str, default='data', help='please input image path')
