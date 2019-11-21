import numpy as np
from keras import Model
from skimage import transform, io

from misc_utils.prediction_utils import cyclic_stacking, inv_sigmoid
from models import backbone


def softmax(x):
    e_x = np.exp(x - np.max(x))
    return e_x / np.sum(e_x,axis=1,keepdims=True)



def get_resize_image_np(image_path,output_size):
    # 得到改变形状的图片
    image = io.imread(image_path)
    # 转换为numpy
    image_np = np.asarray(image)
    # 得到图片宽、高、通道书
    W, H, channel = image_np.shape
    print(W,H)
    # 将图片大小转换为(output_size,output_size)
    resize_image = transform.resize(image, (output_size, output_size),
                                    order=1, mode='constant',
                                    cval=0, clip=True,
                                    preserve_range=True,
                                    anti_aliasing=True)
    resize_image_np = np.stack([resize_image,]).astype(np.uint8)
    print(resize_image_np.shape)
    return resize_image_np,W,H

def task3_tta_predict(model, img_arr):
    img_arr_tta = cyclic_stacking(img_arr)
    pred_logits = np.zeros(shape=(img_arr.shape[0], 7))

    for _img_crops in img_arr_tta:
        pred_logits += model.predict(_img_crops)

    pred_logits = pred_logits/len(img_arr_tta)

    return pred_logits

def predict(image_path):
    num_folds =5

    use_tta = False

    y_pred = np.zeros(shape=(1,7))

    resize_image_np,W,H = get_resize_image_np(image_path,224)

    for k_fold in range(num_folds):
        print("Processing fold ",k_fold)
        run_name = "task3_inception_v3_k"+str(k_fold)+"_v0"
        model = backbone("inception_v3").classification_model(load_from=run_name)
        # 这里可以考虑使用多线程
        predictions_model = Model(inputs=model.input,outputs=model.get_layer('predictions').output)
        if use_tta:
            y_pred += inv_sigmoid(task3_tta_predict(model=predictions_model,img_arr=resize_image_np))
        else:
            y_pred += inv_sigmoid(model.predict_on_batch(resize_image_np))
    y_pred = y_pred / num_folds
    y_prob = softmax(y_pred)
    print('image,MEL,NV,BCC,AKIEC,BKL,DF,VASC\n')
    print(y_prob)

if __name__ == '__main__':
    predict("/home/zhangfan/workData/LinuxCode/pythonProject/ISIC2018/datasets/ISIC2018/data/ISIC2018_Task3_Training_Input/ISIC_0024306.jpg")










