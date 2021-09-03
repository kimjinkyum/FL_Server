# coding=utf-8
import numpy as np
import json
from options import args_parser
import requests
from models import *
import pickle
import torch
from io import BytesIO
from utils import *
import logging

logger = logging.getLogger('Communication')

logger.setLevel(logging.INFO)
formatter = logging.Formatter('%(asctime)s - %(name)s - %(message)s')

stream_handler = logging.StreamHandler()
stream_handler.setFormatter(formatter)
logger.addHandler(stream_handler)


class FL_server:
    def __init__(self):
        self.url = ""
        self.args = args_parser()
        self.urls = ""
        self.id = ""
        self.param = {}
        self.global_model = select_model(self.args)
        self.flag = True
        self.weight = ""
        self.train_loss = ""

    def initialize(self, id, url):
        self.id = id
        self.args["id"] = id
        self.urls = "http://" + url + ':8585/'

        send_url = self.urls + "init"
        logger.info('Send to client id {}'.format(self.id))
        res = requests.post(send_url, data=json.dumps(self.args))
        return res.text

    def start(self, server_global_model):
        logger.info("Start Training client id {}".format(self.id))
        self.global_model = server_global_model
        self.send_weight()
        self.receive_weight()
        return self.weight

    def send_weight(self):
        self.global_model.to('cuda')
        self.weight = self.global_model.state_dict()

        files = write_weight(self.weight)
        logger.info("Send weight to client id {}".format(self.id))

        send_url = self.urls + "download"
        res = requests.post(send_url, files=files)
        self.train_loss = res.text
        print(self.train_loss)
        return None

    def receive_weight(self):
        send_url = self.urls + "update"
        res = requests.post(send_url)
        file_names = 'client_model' + str(self.id) + ".pkl"
        with open(file_names, 'wb') as f:
            for chunk in res.iter_content(chunk_size=128):
                f.write(chunk)
        print("----------------------", file_names)


def get_optimal(arg, param):
    # TODO: Get optimal
    # print(param['local_ep'])
    arg['local_ep'] = 10 * param['local_ep']

    return arg


def select_model(args):
    if args['dataset'] == 'mnist':
        global_model = CNNMnist(args=args)
    elif args['dataset'] == 'cifar':
        global_model = CNNCifar(args=args)

    return global_model.to('cuda')


def write_weight(global_weights):
    file_name = "initial_global_model.pth"

    # copy weights & Save weights
    torch.save(global_weights, file_name)
    data = {'file_name': file_name}

    files = {
        'json': ('json_data', json.dumps(data), 'application/json'),
        'model': (file_name, open(file_name, "rb"), 'application/octet-stream')}

    return files
