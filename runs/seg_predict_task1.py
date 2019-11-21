import numpy
from PIL import Image
from keras.models import load_model, model_from_json
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



def get_resize_image_np(image_path,output_size):
    # 得到改变形状的图片
    image = io.imread(image_path)
    # 转换为numpy
    image_np = numpy.asarray(image)
    # 得到图片宽、高、通道书
    W, H, channel = image_np.shape
    print(W,H)
    # 将图片大小转换为(output_size,output_size)
    resize_image = transform.resize(image, (output_size, output_size),
                                     order=1, mode='constant',
                                     cval=0, clip=True,
                                     preserve_range=True,
                                     anti_aliasing=True)
    resize_image_np = numpy.stack([resize_image,]).astype(numpy.uint8)
    print(resize_image_np.shape)
    return resize_image_np,W,H



def predict(image_path):
    num_folds = 1
    use_tta = False
    y_pred = numpy.zeros(shape=(1,224,224))
    resize_image_np, W, H = get_resize_image_np(image_path, 224)
    for k_fold in range(num_folds):
        print('Processing fold ' , k_fold)
        model_name = 'task1_vgg16'
        run_name = 'task1_vgg16_k%d_v0' % (k_fold)
        model = backbone('vgg16').segmentation_model(load_from = run_name)
        model.load_weights("/home/zhangfan/workData/LinuxCode/pythonProject/ISIC2018/model_data/task1_vgg16_k0_v0/task1_vgg16_k0_v0.hdf5")
        y_pred += inv_sigmoid(model.predict_on_batch(resize_image_np))[:,:,:,0]
    y_pred = y_pred / num_folds
    y_pred = sigmoid(y_pred)
    current_pred = y_pred[0] * 255

    resized_pred = transform.resize(current_pred,output_shape=(W,H),
                                preserve_range=True,
                                    mode='reflect',
                                    anti_aliasing=True)
    resized_pred[resized_pred > 128] = 255
    resized_pred[resized_pred <= 128] = 0
    im = Image.fromarray(resized_pred.astype(numpy.uint8))
    print(im.size)
    im.save("t.png")




if __name__ == '__main__':
    predict("/home/zhangfan/workData/LinuxCode/pythonProject/ISIC2018/datasets/ISIC2018/data/ISIC2018_Task1-2_Test_Input/ISIC_0012236.jpg")


