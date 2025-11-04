import logging
import signal

import tornado.ioloop
import tornado.websocket

from server import make_app


def sig_handler(sig, frame):
    logging.warning('捕获信号: %s', sig)
    io_loop = tornado.ioloop.IOLoop.current()
    io_loop.add_callback_from_signal(io_loop.stop)


if __name__ == "__main__":
    app = make_app()
    app.listen(8888)

    signal.signal(signal.SIGTERM, sig_handler)
    signal.signal(signal.SIGINT, sig_handler)

    print("服务启动: ws://localhost:8888/video")
    tornado.ioloop.IOLoop.current().start()
    print("服务停止")
