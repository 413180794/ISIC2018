# -*- coding: utf-8 -*-
import json
# 调用python安装的thrift依赖包
from thrift.transport import TSocket
from thrift.transport import TTransport
from thrift.protocol import TBinaryProtocol
from thrift.server import TServer
# 调用win10下thrift命令自动生成的依赖包；
# 自动生成的依赖包名为gen-py，但是会报错；
# 所以手动把包名改成gen_py运行成功，这是为什么？
from thrift_py.server import printService


class TestHandler:
    def print_msg(self, msg):
        try:
            # nodejs传过来的字符串数据
            print(msg, type(msg))
            # 把字符串数据转成字典格式
            msg = json.loads(msg)
            print(msg, type(msg))
            # 在字典中添加一个键值对
            msg['email'] = 'yun@23.com'
            # 把字典转成字符串后返回给nodejs
            return json.dumps(msg)
        except Exception as e:
            print(e)
            # 把异常错误写成字典格式
            result = {'errorType': e}
            # 把字典转成字符串后返回给nodejs
            return json.dumps(result)


# 创建服务端
processor = printService.Processor(TestHandler())
# 监听端口
transport = TSocket.TServerSocket(host='127.0.0.1', port=8080)
# 选择传输层
tfactory = TTransport.TBufferedTransportFactory()
# 选择传输协议
pfactory = TBinaryProtocol.TBinaryProtocolFactory()
# 创建服务端
server = TServer.TSimpleServer(processor, transport, tfactory, pfactory)
print("Starting thrift server in python...")
server.serve()