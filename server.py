# -*- utf8 -*-
'''
本代码目的是利用socket.io实现服务端，构建NodeJs与Python代码交互的桥梁

构想：
不需要room的概念，目前实现是一个客户端对应一个服务，不支持多开（多开也没有意义）
这里会在开启软件的时候检测是否已经打开


'''
import os
import numpy as np
import torch
from aiohttp import web

import socketio
from models import backbone
from paths import model_data_dir

sio = socketio.AsyncServer(async_mode='aiohttp',cors_allowed_origins="*")
app = web.Application()
sio.attach(app)

# 预先载入三个模型
# task1_model_name = 'task1_vgg16_k4_v0'
# task1_model = backbone('vgg16').segmentation_model(load_from=task1_model_name)
# task2_model_name = os.path.join(model_data_dir,'task2_vgg16_k0_v0','task2_vgg16_k0_v0.ckpt')
# task2_model = torch.load(task2_model_name,map_location='cpu')
# task3_model_name = 'task3_inception_v3_k4_v0'
# task3_model = backbone('inception_v3').classification_model(load_from=task3_model_name)

async def background_task():
    """后台运行的任务"""
    pass
    # count = 0
    # while True:
        # await sio.sleep(10)
        # count += 1
        # await sio.emit('my_response', {'data': 'Server generated event'})


async def index(request):
    pass

@sio.event
async def seg_predict_image_task1(sid,message):
    '''
    预测一张图像，返回图像的矩阵，与图像的长宽
    :param sid: sid
    :param message: message中包含图片的一维矩阵（RGBA），以及(R,G,B,A)
    :return:
    '''
    image_array = message['image_np'] # 获得图片的一维矩阵，由于canvas中getImageData得到是图片RGBA矩阵，这里默认图片是RGBA格式
    WIDTH = message['width'] # 获得图片的宽
    HEIGHT = message['height'] # 获得图片的长
    image_np = np.asarray(image_array) # 得到图片的numpy对象
    image_np = image_np.reshape(image_np,(WIDTH,HEIGHT,3,1)) #



@sio.event
async def my_event(sid, message):
    print(message)
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
    await sio.emit("connect_success",{'data':"连接成功"},room=sid)
    # await sio.emit('my_response', {'data': 'Connected', 'count': 0}, room=sid)


@sio.event
def disconnect(sid):
    print('Client disconnected')


app.router.add_get('/', index)


if __name__ == '__main__':
    sio.start_background_task(background_task)
    web.run_app(app)