/*
thrift接口定义文件
*/

/*
string print_msg(1:string msg)中的print_msg即为服务端中等待被调用的函数
括号中的1:string msg表示传入的参数为字符串格式，外层的string表示传出的数据为字符串格式
*/
service printService {
    string print_msg(1:string msg)
}