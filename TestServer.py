import os
import unittest

# 编写服务器单元测试
from array import array

import thriftpy2

from thriftServer import Handler

predict_thrift = thriftpy2.load("server.thrift", module_name="predict_thrift")

class MyTestCase(unittest.TestCase):

    def setUp(self) -> None:
        self.handle = Handler()  # 加载深度学习
        self.task1_2_pic_1_path = "/home/zhangfan/workData/LinuxCode/pythonProject/ISIC2018/datasets/ISIC2018/data/ISIC2018_Task1-2_Test_Input/ISIC_0012169.jpg"
        self.task1_2_pic_2_path = "/home/zhangfan/workData/LinuxCode/pythonProject/ISIC2018/datasets/ISIC2018/data/ISIC2018_Task1-2_Test_Input/ISIC_0012236.jpg"
        self.task1_2_pic_3_path = "/home/zhangfan/workData/LinuxCode/pythonProject/ISIC2018/datasets/ISIC2018/data/ISIC2018_Task1-2_Test_Input/ISIC_0012292.jpg"
        self.task1_2_pic_4_path = "/home/zhangfan/workData/LinuxCode/pythonProject/ISIC2018/datasets/ISIC2018/data/ISIC2018_Task1-2_Test_Input/ISIC_0012292.jpg"
        self.task1_2_pic_5_path = "/home/zhangfan/workData/LinuxCode/pythonProject/ISIC2018/datasets/ISIC2018/data/ISIC2018_Task1-2_Test_Input/ISIC_0012302.jpg"

        self.task3_pic_1_path = "/home/zhangfan/workData/LinuxCode/pythonProject/ISIC2018/datasets/ISIC2018/data/ISIC2018_Task3_Test_Input/ISIC_0034524.jpg"
        self.task3_pic_2_path = "/home/zhangfan/workData/LinuxCode/pythonProject/ISIC2018/datasets/ISIC2018/data/ISIC2018_Task3_Test_Input/ISIC_0034525.jpg"
        self.task3_pic_3_path = "/home/zhangfan/workData/LinuxCode/pythonProject/ISIC2018/datasets/ISIC2018/data/ISIC2018_Task3_Test_Input/ISIC_0034526.jpg"
        self.task3_pic_4_path = "/home/zhangfan/workData/LinuxCode/pythonProject/ISIC2018/datasets/ISIC2018/data/ISIC2018_Task3_Test_Input/ISIC_0034527.jpg"
        self.task3_pic_5_path = "/home/zhangfan/workData/LinuxCode/pythonProject/ISIC2018/datasets/ISIC2018/data/ISIC2018_Task3_Test_Input/ISIC_0034528.jpg"

        self.task4_pic_benign_1_path = "/home/zhangfan/workData/LinuxCode/pythonProject/ISIC2018/datasets/ISIC2018/data/ISIC2018_Task4_Test_Input/benign/1.jpg"
        self.task4_pic_benign_2_path = "/home/zhangfan/workData/LinuxCode/pythonProject/ISIC2018/datasets/ISIC2018/data/ISIC2018_Task4_Test_Input/benign/2.jpg"
        self.task4_pic_benign_3_path = "/home/zhangfan/workData/LinuxCode/pythonProject/ISIC2018/datasets/ISIC2018/data/ISIC2018_Task4_Test_Input/benign/5.jpg"
        self.task4_pic_benign_4_path = "/home/zhangfan/workData/LinuxCode/pythonProject/ISIC2018/datasets/ISIC2018/data/ISIC2018_Task4_Test_Input/benign/8.jpg"
        self.task4_pic_benign_5_path = "/home/zhangfan/workData/LinuxCode/pythonProject/ISIC2018/datasets/ISIC2018/data/ISIC2018_Task4_Test_Input/benign/9.jpg"

        self.task4_pic_malignant_1_path = "/home/zhangfan/workData/LinuxCode/pythonProject/ISIC2018/datasets/ISIC2018/data/ISIC2018_Task4_Test_Input/malignant/1.jpg"
        self.task4_pic_malignant_2_path = "/home/zhangfan/workData/LinuxCode/pythonProject/ISIC2018/datasets/ISIC2018/data/ISIC2018_Task4_Test_Input/malignant/3.jpg"
        self.task4_pic_malignant_3_path = "/home/zhangfan/workData/LinuxCode/pythonProject/ISIC2018/datasets/ISIC2018/data/ISIC2018_Task4_Test_Input/malignant/4.jpg"
        self.task4_pic_malignant_4_path = "/home/zhangfan/workData/LinuxCode/pythonProject/ISIC2018/datasets/ISIC2018/data/ISIC2018_Task4_Test_Input/malignant/8.jpg"
        self.task4_pic_malignant_5_path = "/home/zhangfan/workData/LinuxCode/pythonProject/ISIC2018/datasets/ISIC2018/data/ISIC2018_Task4_Test_Input/malignant/13.jpg"

    def test_task1_one_pic(self):
        '''
        测试任务一
        只输入一张图片
        :return:
        '''
        image_paths = [self.task1_2_pic_1_path]
        results = self.handle.seg_predict_images_task1(image_paths)

        self.assertListEqual(results, [
            '/home/zhangfan/workData/LinuxCode/pythonProject/ISIC2018/task1_result/ISIC_0012169_result.jpg'])
        for path in results:
            self.assertTrue(os.path.exists(path))

    def test_task1_many_pic(self):
        '''
        测试任务一
        输入五张图片
        :return:
        '''
        image_paths = [
            self.task1_2_pic_1_path,
            self.task1_2_pic_2_path,
            self.task1_2_pic_3_path,
            self.task1_2_pic_4_path,
            self.task1_2_pic_5_path,
        ]
        results = self.handle.seg_predict_images_task1(image_paths)
        print(results)
        for path in results:
            self.assertTrue(os.path.exists(path))

    def test_task2_one_pic(self):
        image_paths = [self.task1_2_pic_1_path]
        results = self.handle.seg_predict_images_task2(image_paths)
        print(results)
        for path in results:
            self.assertTrue(os.path.exists(path))

    def test_task2_many_pic(self):
        image_paths = [
            self.task1_2_pic_1_path,
            self.task1_2_pic_2_path,
            self.task1_2_pic_3_path,
            self.task1_2_pic_4_path,
            self.task1_2_pic_5_path,
        ]
        results = self.handle.seg_predict_images_task2(image_paths)
        # print(results)
        self.assertListEqual(results, [
            '/home/zhangfan/workData/LinuxCode/pythonProject/ISIC2018/task2_result/ISIC_0012169_pigment_network.jpg',
            '/home/zhangfan/workData/LinuxCode/pythonProject/ISIC2018/task2_result/ISIC_0012169_negative_network.jpg',
            '/home/zhangfan/workData/LinuxCode/pythonProject/ISIC2018/task2_result/ISIC_0012169_streaks.jpg',
            '/home/zhangfan/workData/LinuxCode/pythonProject/ISIC2018/task2_result/ISIC_0012169_milia_like_cyst.jpg',
            '/home/zhangfan/workData/LinuxCode/pythonProject/ISIC2018/task2_result/ISIC_0012169_globules.jpg',
            '/home/zhangfan/workData/LinuxCode/pythonProject/ISIC2018/task2_result/ISIC_0012169_result.jpg',
            '/home/zhangfan/workData/LinuxCode/pythonProject/ISIC2018/task2_result/ISIC_0012236_pigment_network.jpg',
            '/home/zhangfan/workData/LinuxCode/pythonProject/ISIC2018/task2_result/ISIC_0012236_negative_network.jpg',
            '/home/zhangfan/workData/LinuxCode/pythonProject/ISIC2018/task2_result/ISIC_0012236_streaks.jpg',
            '/home/zhangfan/workData/LinuxCode/pythonProject/ISIC2018/task2_result/ISIC_0012236_milia_like_cyst.jpg',
            '/home/zhangfan/workData/LinuxCode/pythonProject/ISIC2018/task2_result/ISIC_0012236_globules.jpg',
            '/home/zhangfan/workData/LinuxCode/pythonProject/ISIC2018/task2_result/ISIC_0012236_result.jpg',
            '/home/zhangfan/workData/LinuxCode/pythonProject/ISIC2018/task2_result/ISIC_0012292_pigment_network.jpg',
            '/home/zhangfan/workData/LinuxCode/pythonProject/ISIC2018/task2_result/ISIC_0012292_negative_network.jpg',
            '/home/zhangfan/workData/LinuxCode/pythonProject/ISIC2018/task2_result/ISIC_0012292_streaks.jpg',
            '/home/zhangfan/workData/LinuxCode/pythonProject/ISIC2018/task2_result/ISIC_0012292_milia_like_cyst.jpg',
            '/home/zhangfan/workData/LinuxCode/pythonProject/ISIC2018/task2_result/ISIC_0012292_globules.jpg',
            '/home/zhangfan/workData/LinuxCode/pythonProject/ISIC2018/task2_result/ISIC_0012292_result.jpg',
            '/home/zhangfan/workData/LinuxCode/pythonProject/ISIC2018/task2_result/ISIC_0012292_pigment_network.jpg',
            '/home/zhangfan/workData/LinuxCode/pythonProject/ISIC2018/task2_result/ISIC_0012292_negative_network.jpg',
            '/home/zhangfan/workData/LinuxCode/pythonProject/ISIC2018/task2_result/ISIC_0012292_streaks.jpg',
            '/home/zhangfan/workData/LinuxCode/pythonProject/ISIC2018/task2_result/ISIC_0012292_milia_like_cyst.jpg',
            '/home/zhangfan/workData/LinuxCode/pythonProject/ISIC2018/task2_result/ISIC_0012292_globules.jpg',
            '/home/zhangfan/workData/LinuxCode/pythonProject/ISIC2018/task2_result/ISIC_0012292_result.jpg',
            '/home/zhangfan/workData/LinuxCode/pythonProject/ISIC2018/task2_result/ISIC_0012302_pigment_network.jpg',
            '/home/zhangfan/workData/LinuxCode/pythonProject/ISIC2018/task2_result/ISIC_0012302_negative_network.jpg',
            '/home/zhangfan/workData/LinuxCode/pythonProject/ISIC2018/task2_result/ISIC_0012302_streaks.jpg',
            '/home/zhangfan/workData/LinuxCode/pythonProject/ISIC2018/task2_result/ISIC_0012302_milia_like_cyst.jpg',
            '/home/zhangfan/workData/LinuxCode/pythonProject/ISIC2018/task2_result/ISIC_0012302_globules.jpg',
            '/home/zhangfan/workData/LinuxCode/pythonProject/ISIC2018/task2_result/ISIC_0012302_result.jpg']
                             )
        for path in results:
            self.assertTrue(os.path.exists(path))

    def test_task3_one_pic(self):
        image_paths = [
            self.task3_pic_1_path,
        ]
        results = self.handle.cls_predict_images_task3(image_paths)
        print(results)

    def test_task3_many_pic(self):
        image_paths = [
            self.task3_pic_1_path,
            self.task3_pic_2_path,
            self.task3_pic_3_path,
            self.task3_pic_4_path,
            self.task3_pic_5_path,
        ]
        results = self.handle.cls_predict_images_task3(image_paths)
        print(results)

    def test_task4_one_benign_pic(self):
        image_paths = [
            self.task4_pic_benign_1_path
        ]
        results = self.handle.cls_predict_images_task4(image_paths)
        print(results)

    def test_task4_on_malignant_pic(self):
        image_paths = [
            self.task4_pic_malignant_1_path
        ]
        results = self.handle.cls_predict_images_task4(image_paths)
        print(results)

    def test_task4_many_pic(self):
        image_paths = [
            self.task4_pic_benign_1_path,
            self.task4_pic_benign_2_path,
            self.task4_pic_benign_3_path,
            self.task4_pic_benign_4_path,
            self.task4_pic_benign_5_path,
            self.task4_pic_malignant_1_path,
            self.task4_pic_malignant_2_path,
            self.task4_pic_malignant_3_path,
            self.task4_pic_malignant_4_path,
            self.task4_pic_malignant_5_path,
        ]
        results = self.handle.cls_predict_images_task4(image_paths)
        print(results)


if __name__ == '__main__':
    unittest.main()
