import os
import inspect


def mkdir_if_not_exist(dir_list):
    for directory in dir_list:
        if not os.path.exists(directory):
            os.makedirs(directory)


curr_filename = inspect.getfile(inspect.currentframe())
root_dir = os.path.dirname(os.path.abspath(curr_filename))
task1_result_dir = os.path.join(root_dir,"task1_result")
task2_result_dir = os.path.join(root_dir,"task2_result")

model_data_dir = os.path.join(root_dir, 'model_data')
submission_dir = os.path.join(root_dir, 'submissions')
task1_model_name = 'task1_vgg16_k4_v0'
task2_model_name = os.path.join(model_data_dir, 'task2_vgg16_k0_v0', 'task2_vgg16_k0_v0.ckpt')
task3_model_name = 'task3_inception_v3_k4_v0'
task4_model_name = 'task4_inception_v3_k4_v0'
dir_to_make = [model_data_dir, submission_dir,task1_result_dir,task2_result_dir]
mkdir_if_not_exist(dir_list=dir_to_make)
mkdir_if_not_exist(dir_list=dir_to_make)
