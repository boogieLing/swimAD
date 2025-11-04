import asyncio
import websockets


async def websocket_client():
    uri = "ws://localhost:8888/alert"
    async with websockets.connect(uri) as websocket:
        # 初始发送消息
        await websocket.send("Hello, Server!")
        print(f"客户端发送: Hello, Server!")

        # 持续接收消息
        while True:
            try:
                response = await websocket.recv()
                print(f"服务器回复: {response}")
            except websockets.exceptions.ConnectionClosed:
                print("连接已关闭")
                break
            except Exception as e:
                print(f"发生错误: {e}")
                break


if __name__ == "__main__":
    asyncio.run(websocket_client())