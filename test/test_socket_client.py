import argparse
import socket
import json
import time
import random

class SocketClient:
    def __init__(self, server_ip: str, server_port: int, max_retries=3):
        self.server_ip = server_ip
        self.server_port = server_port
        self.client_socket = None
        self.connected = False
        self.max_retries = max_retries

    def connect(self):
        retries = 0
        while retries < self.max_retries:
            try:
                self.client_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                self.client_socket.connect((self.server_ip, self.server_port))
                self.connected = True
                print(f"已连接服务器({self.server_ip}:{self.server_port})")
                return True
            except Exception as e:
                retries += 1
                print(f"第 {retries} 次连接服务器时出现异常: {e}")
                if retries < self.max_retries:
                    print("正在重试...")
                else:
                    print("达到最大重试次数，连接失败")
                    self.close()
        return False

    def send_identity(self, identity: str):
        if not self.connected:
            print("未连接到服务器，无法发送数据")
            return
        try:
            self.client_socket.sendall(identity.encode())
        except Exception as e:
            print(f"发送身份信息时出现异常: {e}")

    def send_net_features(self, net_features: dict):
        if not self.connected:
            print("未连接到服务器，无法发送数据")
            return
        try:
            json_data = json.dumps(net_features)
            self.client_socket.sendall(json_data.encode())
        except Exception as e:
            print(f"发送数据时出现异常: {e}")

    def receive_intercept_strategy(self, buffer_size=1024, close_on_empty=True):
        if not self.connected:
            print("未连接到服务器，无法接收数据")
            return None
        try:
            data = self.client_socket.recv(buffer_size)
            if not data:
                print("未接收到数据，可能服务器已关闭连接")
                self.connected = False
                if close_on_empty:
                    self.close()
                return None
            decoded_data = data.decode()
            try:
                json_obj = json.loads(decoded_data)
                return list(json_obj)
            except json.JSONDecodeError:
                print(f"接收到的数据不是有效的 JSON 格式: {decoded_data}")
                return None
        except UnicodeDecodeError:
            print("无法解码接收到的数据")
            return None
        except Exception as e:
            print(f"接收数据时出现异常: {e}")
            return None

    def close(self):
        try:
            if self.client_socket:
                self.client_socket.close()
                self.connected = False
                print(f"已关闭与服务器({self.server_ip}:{self.server_port})的连接")
        except Exception as e:
            print(f"关闭客户端连接时出现异常: {e}")


if __name__ == '__main__':
    SERVER_IP = '10.129'
    SERVER_PORT = 65432
    IDENTITYS = ['webrtc_sender', 'base_station']
    socket_client = SocketClient(SERVER_IP, SERVER_PORT)
    socket_client.connect()
    
    parser = argparse.ArgumentParser(description='Socket Client')
    parser.add_argument('--identity', type=str, default=1, choices=IDENTITYS,
                    help='身份 (webrtc_sender or base_station)')
    args = parser.parse_args()
    identity = args.identity
    print(f'当前客户端身份: {identity}')
    socket_client.send_identity(identity)

    try:
        while True:
            net_features = {
                "loss": 0.1,
                "rtt": 100,
                "throughput": 1000,
                "available_resources": 0.5,
                "tc_throughput": 1200
            }
            print("发送的网络特征:", net_features)
            socket_client.send_net_features(net_features)
            intercept_strategy = socket_client.receive_intercept_strategy()

            time.sleep(random.randint(5, 15)/10)

            if intercept_strategy is None:
                break
            print("接收到的拦截策略:", intercept_strategy)
    except KeyboardInterrupt:
        print("用户手动终止程序，正在关闭连接...")
    except Exception as e:
        print(f"程序出现异常: {e}，正在关闭连接...")
    finally:
        socket_client.close()