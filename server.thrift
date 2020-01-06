/*
thrift接口定义文件
*/

/*
string print_msg(1:string msg)中的print_msg即为服务端中等待被调用的函数
括号中的1:string msg表示传入的参数为字符串格式，外层的string表示传出的数据为字符串格式

python为Nodejs提供深度学习的函数调用
*/
struct task3_result{
    1:string name;
    2:list<double> result;
}

struct task4_result{
    1:string name;
    2:list<double> result;
}
service predictService {
// 注意地址的应用，我们不保存图片，因为图片已经保存在本地了
// 稍后做如下优化
// 将识别结果的图片地址保存在内存中
// 如果该地址已经在内存中，说明已经识别过了，再去内存读取该图片，如果内存中没有，则去硬盘中读，从硬盘中读取后缓存在内存中
//  task1应用：识别一组图片，返回一组图片识别结果的地址
    list<string> seg_predict_images_task1(1:list<string> image_paths)

//  task2应用：识别一组图片，返回一组结果，见上
    list<string> seg_predict_images_task2(1:list<string> image_paths)

//  task3应用：分类一组图片
    list<task3_result> cls_predict_images_task3(1:list<string> image_paths)

//  task4应用:二分类一组图片
    list<task4_result> cls_predict_images_task4(1:list<string> image_paths)
}