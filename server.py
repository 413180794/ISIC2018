# -*- utf8 -*-
'''
本代码目的是利用socket.io实现服务端，构建NodeJs与Python代码交互的桥梁

构想：
不需要room的概念，目前实现是一个客户端对应一个服务，不支持多开（多开也没有意义）
这里会在开启软件的时候检测是否已经打开

TODO：测试客户端canvas传输而来的图像矩阵
'''
import os
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import torch
from aiohttp import web

import socketio
from skimage import transform
from skimage.color import rgba2rgb

from misc_utils.prediction_utils import inv_sigmoid, sigmoid
from models import backbone
from paths import model_data_dir

sio = socketio.AsyncServer(async_mode='aiohttp', cors_allowed_origins="*")
app = web.Application()
sio.attach(app)

# 预先载入三个模型
TASK1_MODEL_NAME = 'task1_vgg16_k4_v0'
TASK2_MODEL_NAME = os.path.join(model_data_dir, 'task2_vgg16_k0_v0', 'task2_vgg16_k0_v0.ckpt')
TASK3_MODEL_NAME = 'task3_inception_v3_k4_v0'
task1_model = None
task2_model = None
task3_model = None
load_model_success = False


def load_model():
    '''加载模型函数'''
    global task1_model
    global task2_model
    global task3_model
    task1_model = backbone('vgg16').segmentation_model(load_from=TASK1_MODEL_NAME)
    task2_model = torch.load(TASK2_MODEL_NAME, map_location='cpu')
    task3_model = backbone('inception_v3').classification_model(load_from=TASK3_MODEL_NAME)


load_model()
load_model_success = True


async def background_task():
    """后台运行的任务"""
    # global load_model_success
    # loop = app.loop
    # executor = ThreadPoolExecutor(4)
    # await loop.run_in_executor(executor, load_model)
    # load_model_success = True  # 成功加载模型

    # count = 0
    # while True:
    # await sio.sleep(10)
    # count += 1
    # await sio.emit('my_response', {'data': 'Server generated event'})


async def index(request):
    pass


@sio.event
async def seg_predict_image_task1(sid, message):
    '''
    预测一张图像，返回图像的矩阵，与图像的长宽
    :param sid: sid
    :param message: message中包含图片的一维矩阵（RGBA），以及(R,G,B,A)
    :return:
    '''
    print(message)
    # 获得图片的一维矩阵，由于canvas中getImageData得到是图片RGBA矩阵，这里默认图片是RGBA格式
    image_array, width, height = message['image_np'], message['width'], message['height']
    # 得到图片的numpy对象,并将canvas生成的rgba矩阵转为rgb矩阵
    image_np = rgba2rgb(np.reshape(np.asarray(image_array), (width, height, 4)))
    # y_pred = np.zeros(shape=(1, 244, 244))
    # 取得供深度学习处理的图像矩阵
    resize_image_np = np.stack(
        [transform.resize(image_np, (244, 244), order=1, mode='constant', cval=0, clip=True,
                          preserve_range=True), ]).astype(np.uint8)
    y_pred = inv_sigmoid(task1_model.predict_on_batch(resize_image_np))[:, :, :, 0]
    y_pred = sigmoid(y_pred)
    current_pred = y_pred[0] * 255
    current_pred[current_pred > 128] = 255
    current_pred[current_pred <= 128] = 0
    print(current_pred.shape)
    # 将图像还原为原来的大小
    resized_pred = transform.resize(current_pred, output_shape=(width, height),
                                    preserve_range=True,
                                    mode='reflect')

    # 将图像转换为rgba矩阵
    print(resized_pred)

    # await sio.emit("single_task1_result",{""})
    return current_pred


@sio.event
async def my_event(sid, message):
    print(message)
    image_array = message['data']
    width = message['width']
    height = message['height']
    image_np = np.asarray(image_array).reshape((width, height, 4))
    print(image_np.shape)
    image_np = rgba2rgb(image_np)
    print(image_np.shape)

    await sio.emit('my_response', {'data': message['data']}, room=sid)


@sio.event
async def my_broadcast_event(sid, message):
    await sio.emit('my_response', {'data': message['data']})


@sio.event
async def join(sid, message):
    sio.enter_room(sid, message['room'])
    await sio.emit('my_response', {'data': 'Entered room: ' + message['room']},
                   room=sid)


@sio.event
async def leave(sid, message):
    sio.leave_room(sid, message['room'])
    await sio.emit('my_response', {'data': 'Left room: ' + message['room']},
                   room=sid)


@sio.event
async def close_room(sid, message):
    await sio.emit('my_response',
                   {'data': 'Room ' + message['room'] + ' is closing.'},
                   room=message['room'])
    await sio.close_room(message['room'])


@sio.event
async def my_room_event(sid, message):
    await sio.emit('my_response', {'data': message['data']},
                   room=message['room'])


@sio.event
async def disconnect_request(sid):
    await sio.disconnect(sid)


@sio.event
async def connect(sid, environ):
    # 告知客户端连接成功
    await sio.emit("connect_success", {'data': "连接成功"}, room=sid)
    while not load_model_success:
        await sio.sleep(1)
    await sio.emit('load_model_success', {'data': "载入模型成功"}, room=sid)
    # await sio.emit('my_response', {'data': 'Connected', 'count': 0}, room=sid)


@sio.event
def disconnect(sid):
    print('Client disconnected')


app.router.add_get('/', index)

if __name__ == '__main__':
    sio.start_background_task(background_task)
    web.run_app(app)
