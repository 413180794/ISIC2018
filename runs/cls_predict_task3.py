import cv2
import numpy as np
from PIL import Image

from misc_utils.prediction_utils import cyclic_stacking, inv_sigmoid
from models import backbone


def softmax(x):
    e_x = np.exp(x - np.max(x))
    return e_x / np.sum(e_x,axis=1,keepdims=True)



def get_resize_image_np(image_path,output_size):
    # 得到改变形状的图片
    image = Image.open(image_path)
    # 转换为numpy
    image_np = np.asarray(image)
    # 得到图片宽、高、通道书
    W, H, channel = image_np.shape
    # 将图片大小转换为(output_size,output_size)
    resize_image_np = cv2.resize(image_np, (output_size, output_size),interpolation=cv2.INTER_CUBIC)
    resize_image_np = np.stack([resize_image_np,]).astype(np.uint8)

    return resize_image_np,W,H

def task3_tta_predict(model, img_arr):
    img_arr_tta = cyclic_stacking(img_arr)
    pred_logits = np.zeros(shape=(img_arr.shape[0], 7))

    for _img_crops in img_arr_tta:
        pred_logits += model.predict(_img_crops)

    pred_logits = pred_logits/len(img_arr_tta)

    return pred_logits

def cls_predict_images(image_paths):
    '''识别一组图片'''
    num_folds = 5
    use_tta = False
    run_name = "task3_inception_v3_k" + str(num_folds) + "_v0"
    model = backbone("inception_v3").classification_model(load_from=run_name)
    for image_path in image_paths:
        y_pred = np.zeros(shape=(1,7))
        resize_image_np, W, H = get_resize_image_np(image_path,224)
        y_pred += inv_sigmoid(model.predict_on_batch(resize_image_np))
        y_prod = softmax(y_pred)
        print({
            "result": y_prod
        })
def cls_predict_image(image_path,use_tta=False):
    '''识别一张图片'''
    num_folds =4

    use_tta = False

    y_pred = np.zeros(shape=(1,7))

    resize_image_np,W,H = get_resize_image_np(image_path,224)
    if use_tta:
        for k_fold in range(num_folds+1):
            run_name = "task3_inception_v3_k"+str(k_fold)+"_v0"
            model = backbone("inception_v3").classification_model(load_from=run_name)
            # 这里可以考虑使用多线程
            if use_tta:
                y_pred += inv_sigmoid(task3_tta_predict(model=model,img_arr=resize_image_np))
            else:
                y_pred += inv_sigmoid(model.predict_on_batch(resize_image_np))
    else:
        run_name = "task3_inception_v3_k" + str(num_folds) + "_v0"
        model = backbone("inception_v3").classification_model(load_from=run_name)
        y_pred += inv_sigmoid(model.predict_on_batch(resize_image_np))

    y_pred = y_pred / num_folds
    y_prob = softmax(y_pred)
    print({
        "result":y_prob
    })

if __name__ == '__main__':
    cls_predict_image("/home/zhangfan/workData/LinuxCode/pythonProject/ISIC2018/datasets/ISIC2018/data/ISIC2018_Task3_Training_Input/ISIC_0024306.jpg")










