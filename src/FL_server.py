# coding=utf-8
import numpy as np
import json
from options import args_parser
import requests
from models import CNNMnist
import pickle
import torch
from io import BytesIO
from utils import *


class FL_server:
    def __init__(self, url):
        self.args = args_parser()
        self.urls = url
        self.param = {}
        self.global_model = CNNMnist(args=self.args)
        self.flag = True
        self.weight = ""

    def initialize(self):
        # TODO : Federated Learning info 초기화 & Sending
        send_url = self.urls + "init"
        res = requests.post(send_url, data=json.dumps(self.args))
        return res.text

    def start(self, id, init):
        if init == "initial":
            if "Success" in self.initialize():
                print("In")
                self.send_weight(init)
                return self.weight
        else:
            self.send_weight(init)
            return self.weight

    def send_weight(self, init):
        if init == "initial":
            self.global_model.to('cuda')
            self.weight = self.global_model.state_dict()

        else:
            # TODO aggregate 받아오기
            pass
        files = write_weight(self.weight)
        send_url = self.urls + "download"
        res = requests.post(send_url, files=files)

        with open('client.pkl', 'wb') as f:
            for chunk in res.iter_content(chunk_size=128):
                f.write(chunk)

        # self.weight = torch.load('client.pkl')
        self.weight = self.global_model.state_dict()
        return None
        # print(self.weight)
        # return self.weight

    def update(self):
        # TODO : Waiting Clients weight
        pass

    def aggregate(self):
        # TODO : Aggregate & download global model
        pass


def get_optimal(arg, param):
    # TODO: Get optimal
    # print(param['local_ep'])
    arg['local_ep'] = 10 * param['local_ep']

    return arg


def write_weight(global_weights):
    file_name = "initial_global_model.pth"

    # copy weights & Save weights
    torch.save(global_weights, file_name)
    data = {'file_name': file_name}

    files = {
        'json': ('json_data', json.dumps(data), 'application/json'),
        'model': (file_name, open(file_name, "rb"), 'application/octet-stream')}

    return files
