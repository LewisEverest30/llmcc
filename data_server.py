import json
import socket
import time
import random


class DataServer:

    def __init__(self, host='0.0.0.0', port=65432, data_queue=None):
        self.data_queue = data_queue
        
        self.host = host
        self.port = port
        self.server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        print("已创建server socket，正在监听连接...")
        self.clients : list[dict] = []
        self.base_station_socket_index : int = None
        self.webrtc_sender_socket_index : int = None

        self.server_socket.bind((self.host, self.port))  # 绑定地址和端口
        print(f"已绑定地址({self.host})和端口({self.port})")
        # self.server_socket.listen(2)  # 监听连接，这里允许的最大连接数设
        # while len(self.clients) < 2:
        self.server_socket.listen(1)  # 监听连接，这里允许的最大连接数设
        while len(self.clients) < 1:
            client_socket, client_address = self.server_socket.accept()# 接受客户端连接，这一步会阻塞直到有客户端连接进来
            print(f"已连接客户端: {client_address}")
            self.clients.append(     # 将客户端套接字和地址保存到列表中
                {
                    "socket": client_socket,
                    "address": client_address
                }
            )

            # identity = self.receive_identity(client_socket)
            # print(f"接收到客户端身份信息: {identity}")
            identity = "webrtc_sender"
            if identity == "base_station":
                self.base_station_socket_index = len(self.clients) - 1
            elif identity == "webrtc_sender":
                self.webrtc_sender_socket_index = len(self.clients) - 1
            else:
                print(f"未知的客户端身份: {identity}")
                raise ValueError(f"未知的客户端身份: {identity}")
            
        self.connected = True

    def receive_identity(self, client_socket):
        """接收客户端发送的身份信息"""
        data = client_socket.recv(1024)  # 假设最大消息大小为1024字节
        return data.decode().strip()    

    def receive_net_features_from_webrtc_sender(self, buffer_size=1024, close_on_empty=True):
        try:
            data_webrtc_sender = self.clients[self.webrtc_sender_socket_index]['socket'].recv(buffer_size)
            if not data_webrtc_sender:
                print("未接收到数据，可能客户端已关闭连接，即将关闭服务器")
                self.connected = False
                if close_on_empty:
                    self.close()
                return None
            decoded_data_webrtc_sender = data_webrtc_sender.decode()
            try:
                json_obj_webrtc_sender = json.loads(decoded_data_webrtc_sender)
                loss = json_obj_webrtc_sender.get("loss")
                rtt = json_obj_webrtc_sender.get("rtt")
                throughput = json_obj_webrtc_sender.get("throughput")

                return rtt, loss, throughput
            except json.JSONDecodeError:
                print(f"接收到的数据不是有效的 JSON 格式: {decoded_data_webrtc_sender}")
                return None
        except UnicodeDecodeError:
            print("无法解码接收到的数据")
            return None
        except Exception as e:
            print(f"接收数据时出现异常: {e}, 客户端{self.clients[self.webrtc_sender_socket_index]['address']}")
            return None


    def receive_net_features_and_available_resources_from_webrtc_sender(self, buffer_size=1024, close_on_empty=True):
        try:
            data_webrtc_sender = self.clients[self.webrtc_sender_socket_index]['socket'].recv(buffer_size)
            if not data_webrtc_sender:
                print("未接收到数据，可能客户端已关闭连接，即将关闭服务器")
                self.connected = False
                if close_on_empty:
                    self.close()
                return None
            decoded_data_webrtc_sender = data_webrtc_sender.decode()
            try:
                json_obj_webrtc_sender = json.loads(decoded_data_webrtc_sender)
                loss = json_obj_webrtc_sender.get("loss")
                rtt = json_obj_webrtc_sender.get("rtt")
                throughput = json_obj_webrtc_sender.get("throughput")
                available_resources = json_obj_webrtc_sender.get("available_resources")

                return rtt, loss, throughput, available_resources
            except json.JSONDecodeError:
                print(f"接收到的数据不是有效的 JSON 格式: {decoded_data_webrtc_sender}")
                return None
        except UnicodeDecodeError:
            print("无法解码接收到的数据")
            return None
        except Exception as e:
            print(f"接收数据时出现异常: {e}, 客户端{self.clients[self.webrtc_sender_socket_index]['address']}")
            return None

    def send_predicted_bitrate_to_webrtc_sender(self, bitrate: float):
        try:
            if not self.connected:
                print("未连接到客户端，无法发送数据")
                return
            json_ret = json.dumps(bitrate)
            self.clients[self.webrtc_sender_socket_index]['socket'].sendall(json_ret.encode())
        except Exception as e:
            print(f"发送预测码率时出现异常: {e}")

    def send_ack_to_webrtc_sender(self, ack_nessage = 'ack'):
        if not self.connected:
            print("未连接到服务器，无法发送数据")
            return
        try:
            self.clients[self.webrtc_sender_socket_index]['socket'].sendall(ack_nessage.encode())
        except Exception as e:
            print(f"向webrtc_sender发送ack时出现异常: {e}")

    def send_ack_to_base_station(self, ack_nessage = 'ack'):
        if not self.connected:
            print("未连接到服务器，无法发送数据")
            return
        try:
            self.clients[self.base_station_socket_index]['socket'].sendall(ack_nessage.encode())
        except Exception as e:
            print(f"向base_station发送ack时出现异常: {e}")

    def receive_ack(self, buffer_size=1024, close_on_empty=True, ack_nessage = 'ack'):
        if not self.connected:
            print("未连接到服务器，无法接收数据")
            return None
        try:
            while True:  # 循环等待基站侧的确认消息，否则阻塞
                data = self.clients[self.base_station_socket_index]['socket'].recv(buffer_size)
                if not data:
                    print("未接收到数据，可能服务器已关闭连接")
                    self.connected = False
                    if close_on_empty:
                        self.close()
                    return None
                decoded_data = data.decode()
                if decoded_data == ack_nessage:
                    return True
                else:
                    continue
        except UnicodeDecodeError:
            print("无法解码接收到的数据")
            return None
        except Exception as e:
            print(f"接收数据时出现异常: {e}, 客户端{self.clients[self.base_station_socket_index]['address']}")

    def close(self):
        try:
            for client in self.clients:
                client['socket'].close()
                print(f"已关闭客户端连接({client['address']})")
            self.connected = False
        except Exception as e:
            print(f"关闭客户端连接时出现异常: {e}")
        try:
            if self.server_socket:
                self.server_socket.close()
                print(f"已关闭服务器套接字({self.server_socket})")
        except Exception as e:
            print(f"关闭服务器套接字时出现异常: {e}")


# if __name__ == '__main__':
    # data_server = DataServer()
    # try:
    #     # while True:
    #     for i in range(5):
    #         data = data_server.receive_net_features()
    #         # if data is None:
    #             # break
    #         print("接收到的网络特征:", data)
            
    #         time.sleep(random.randint(5, 15)/10)

    #         intercept_strategy = [0, 1, 0, 1, 0, 1]
    #         print("发送的拦截策略:", intercept_strategy)
    #         data_server.send_intercept_strategy(intercept_strategy)
    # except KeyboardInterrupt:
    #     print("用户手动终止程序，正在关闭连接...")
    # except Exception as e:
    #     print(f"程序出现异常: {e}，正在关闭连接...")
    # finally:
    #     data_server.close()
