import time
import queue
from scapy.all import IP, sendp
from netfilterqueue import NetfilterQueue
import threading

packet_intercept_buffer = queue.Queue()
lock_for_packet_intercept_buffer = threading.Lock()

def packet_callback(packet):
    # 获取原始数据包
    pkt = IP(packet.get_payload())
    
    # 设置发送时间
    target_time = time.time() + intercept_strategy
    
    with lock_for_packet_intercept_buffer:
        packet_intercept_buffer.put((pkt, target_time))
    
    packet.drop()

def dispatch_due_packets_from_buffer_periodically():
    while True:
        # 获取此批发送的数据包使用的观察时间
        current_time = time.time()
        
        with lock_for_packet_intercept_buffer:
            packets_to_send = []
            while not packet_intercept_buffer.empty():
                pkt, target_time = packet_intercept_buffer.queue[0]
                
                if current_time >= target_time:
                    
                    packets_to_send.append(packet_intercept_buffer.get())  # 移除已发送的数据包并准备发送
                else:
                    break
        
        # 发送数据包
        for pkt, _ in packets_to_send:
            sendp(pkt)
        
        time.sleep(0.1)


