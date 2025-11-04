import json
import tornado

from datetime import datetime
from tornado import gen

from associate import associate
from constant import FREQUENCY
from processor import FrameProcessor
from source import get_source, put_source, del_source

frame_processor: dict[str, FrameProcessor]= {}

class Response:
    def __init__(self, code: int, message: str, data: any):
        self.code = code
        self.msg = message
        self.data = data
        self.time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    def json(self):
        return json.dumps(self.__dict__, ensure_ascii=False)


class VideoWebSocketHandler(tornado.websocket.WebSocketHandler):
    clients = set()  # 存储活跃连接

    def initialize(self):
        self.stop_flag = False

    async def open(self):
        print(f"客户端连接: {self.request.remote_ip}")
        self.clients.add(self)

    async def on_message(self, message):
        print(f"收到消息: {message}")
        tornado.ioloop.IOLoop.current().spawn_callback(self.video_stream_loop, message)

    async def video_stream_loop(self, message):
        processor = frame_processor.get(message, None)
        if processor is None:
            return

        while not self.stop_flag and self in self.clients:
            try:
                # 获取最新处理好的帧
                frame_data = processor.get_current_frame()
                await self.write_message(frame_data, binary=True)
                await gen.sleep(1 / FREQUENCY / 2)  # 控制帧率
            except tornado.websocket.WebSocketClosedError:
                break
            except Exception as e:
                print(f"视频流错误: {e}")
                break

    def on_close(self):
        print("连接关闭")
        self.stop_flag = True
        self.clients.remove(self)

    def check_origin(self, origin):
        return True  # 允许跨域

class AssociateSocketHandler(tornado.websocket.WebSocketHandler):
    clients = set()

    def initialize(self):
        self.stop_flag = False

    async def open(self):
        print(f"客户端连接: {self.request.remote_ip}")
        self.clients.add(self)

    async def on_message(self, message):
        print(f"收到消息: {message}")
        tornado.ioloop.IOLoop.current().spawn_callback(self.associate_loop, message)

    async def associate_loop(self, message):
        while not self.stop_flag and self in self.clients:
            try:
                data = associate.get_associate_result()
                str_data = json.dumps([obj.to_dict() for obj in data.values()])
                await self.write_message(str_data, binary=True)
            except tornado.websocket.WebSocketClosedError:
                break
            except Exception as e:
                print(f"获取融合结果错误: {e}")
                break


    def on_close(self):
        print("连接关闭")
        self.stop_flag = True
        self.clients.remove(self)

    def check_origin(self, origin):
        return True

class AlertSocketHandler(tornado.websocket.WebSocketHandler):
    clients = set()

    def initialize(self):
        self.stop_flag = False

    async def open(self):
        print(f"客户端连接: {self.request.remote_ip}")
        self.clients.add(self)

    async def on_message(self, message):
        print(f"收到消息: {message}")
        tornado.ioloop.IOLoop.current().spawn_callback(self.associate_loop, message)

    async def associate_loop(self, message):
        while not self.stop_flag and self in self.clients:
            try:
                data = associate.get_alert()
                str_data = json.dumps([obj.to_dict() for obj in data])
                await self.write_message(str_data, binary=True)
            except tornado.websocket.WebSocketClosedError:
                break
            except Exception as e:
                print(f"获取告警错误: {e}")
                break


    def on_close(self):
        print("连接关闭")
        self.stop_flag = True
        self.clients.remove(self)

    def check_origin(self, origin):
        return True

class SourceHttpHandler(tornado.web.RequestHandler):
    def get(self):
        self.set_header("Content-Type", "application/json")
        self.write(Response(0, "ok", get_source()).json())

    def put(self):
        try:
            data = json.loads(self.request.body)
            key = data['key']
            src = data['source']
            put_source(key, src)
            if key not in frame_processor.keys():
                frame_processor[src] = FrameProcessor(key, src)
            self.write(Response(0, "ok", "").json())
        except json.JSONDecodeError:
            self.write(Response(400, "Invalid JSON format", "").json())

    def delete(self):
        data = json.loads(self.request.body)
        key = data['key']
        del_source(key)
        if frame_processor[key] is not None:
            frame_processor[key].stop()
            del frame_processor[key]
        self.write(Response(0, "ok", "").json())


def make_app():
    sources = get_source()
    for key in sources.keys():
        frame_processor[key] = FrameProcessor(key, sources[key])

    for key in frame_processor:
        associate.add_stream(key)
        frame_processor[key].start()

    associate.start()

    return tornado.web.Application([
        (r"/video", VideoWebSocketHandler),
        (r"/associate", AssociateSocketHandler),
        (r"/alert", AlertSocketHandler),
        (r"/source", SourceHttpHandler),
    ])
