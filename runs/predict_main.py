import argparse
from runs.cls_predict_task3 import cls_predict_image, cls_predict_images
from runs.seg_predict_task1 import seg_predict_image_task1, seg_predict_images_task1
from runs.seg_predict_task2 import seg_predict_image_task2

signal_task = {
    "task1":seg_predict_image_task1,
    "task2":seg_predict_image_task2,
    "task3":cls_predict_image
}

group_task = {
    "task1":seg_predict_images_task1,
    "task2":seg_predict_image_task2,
    "task3":cls_predict_images
}

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    arg = parser.add_argument
    arg('--image-paths', nargs="+", help='please input image path list')
    arg('--tasks', type=str, nargs="?", choices=['task1','task2','task3'])
    arg('--mode',type=str,nargs="?",choices=['single','group'])
    args = parser.parse_args()
    tasks = args.tasks
    image_paths = args.image_paths
    mode = args.mode
    if not tasks:
        # 如果没有填写任务选项
        raise ValueError("Please specify tasks,choices=['task1','task2','task3']")

    if not image_paths:
        # 如果没有填图片选项
        raise ValueError("No images found.")

    if not mode:
        raise ValueError("Please select mode,single or group")
    # 得到预测函数
    if mode == 'single':
        predict = signal_task[tasks]
    elif mode == 'group':
        predict = group_task[tasks]
    else:
        # 出现异常
        raise ValueError("mode is wrong!")

    predict(image_paths)


