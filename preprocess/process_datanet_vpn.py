import numpy as np
from scapy.compat import raw
from scapy.layers.inet import IP, UDP
from scapy.layers.l2 import Ether
from scapy.packet import Padding
from scipy import sparse
import os
from utlis import should_omit_packet, read_pcap, PREFIX_TO_APP_ID, PREFIX_TO_TRAFFIC_ID
from utlis import PREFIX_TO_TorApp_ID, ID_TO_APP, ID_TO_TRAFFIC
from tqdm import tqdm
import time 
import random




def remove_ether_header(packet):
    if Ether in packet:
        return packet[Ether].payload

    return packet


def mask_ip(packet):
    if IP in packet:
        packet[IP].src = '0.0.0.0'
        packet[IP].dst = '0.0.0.0'

    return packet


def pad_udp(packet):
    if UDP in packet:
        # get layers after udp
        layer_after = packet[UDP].payload.copy()

        # build a padding layer
        pad = Padding()
        pad.load = '\x00' * 12

        layer_before = packet.copy()
        layer_before[UDP].remove_payload()
        packet = layer_before / pad / layer_after

        return packet

    return packet

def packet_to_sparse_array(packet, max_length=1480):
    arr = np.frombuffer(raw(packet), dtype=np.uint8)[0: max_length] #/ 255
    if len(arr) < max_length:
        pad_width = max_length - len(arr)
        arr = np.pad(arr, pad_width=(0, pad_width), constant_values=0)
    return arr


def transform_packet(packet):

    packet = remove_ether_header(packet)
    arr = packet_to_sparse_array(packet)
    if arr is not None:
        token = ""
        for i in arr:
            token = token + " " + transform2hex(i) 
        return token.strip(" ")

def transform_pcap(path):
    
    f_service = open("service/datanet_service.txt",'a')
    f_app = open("app/datanet_app.txt",'a')


    prefix = path.split('/')[-1].split('.')[0].lower()
    app_label = PREFIX_TO_APP_ID.get(prefix)
    service_label = PREFIX_TO_TRAFFIC_ID.get(prefix)


    for i, packet in enumerate(read_pcap(path)):
        token = transform_packet(packet)
        
        if token is not None: 
            if app_label is not None and check(ID_TO_APP[app_label]) is True:
                f_app.write(token+"\t"+str(app_label)+"\n")
          
            if service_label is not None and check(ID_TO_TRAFFIC[service_label]) is True:
                f_service.write(token+"\t"+str(service_label)+"\n")
    

def all_path(dirname):
    result = []
    for maindir, subdir, file_name_list in os.walk(dirname):
        for filename in file_name_list:
            apath = os.path.join(maindir, filename)
            result.append(apath)
    return result

classes = {}
def check(label):
    if label not in classes:
        classes[label] = 1
    elif re[id] <= 50000:
        classes[id] += 1
        return True
    else:
        return False
        
if __name__ == '__main__':

    # source: packet or PCAP orgin file path
    source =  "/home/dl/Desktop/program/Traffic_class/Public_dataset/VPN-Pcaps/packet/" 
    root = os.listdir(source)
    random.shuffle(root)
    for i in tqdm(root):
        path = source + i
        transform_pcap(path)

    
