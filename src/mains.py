# from kubernetes import client, config
import requests

from options import *
from FL_server import FL_server
from concurrent.futures import ThreadPoolExecutor
from utils import *

import logging

# 로그 생성
logger = logging.getLogger("Server")
logger.setLevel(logging.INFO)

# log 출력 형식
formatter = logging.Formatter('%(asctime)s - %(name)s - %(message)s')
stream_handler = logging.StreamHandler()
stream_handler.setFormatter(formatter)
logger.addHandler(stream_handler)

config.load_kube_config()
v1 = client.CoreV1Api()

print("Listing pods with their IPs")

# 쿠버네티스에서 마스터와 연동되어 있는 모든 Pod 가져오기
ret = v1.list_pod_for_all_namespaces(watch=False)
ip_lists = []
application_name = "fl-ex"


for i in ret.items:
    # print("%s\t%s\t%s" % (i.status.pod_ip, i.metadata.namespace, i.metadata.name))

    # 그중에서도 Pod 배포 할 때 지정한 Name 과 동일한 pod IP 리스트 업
    if application_name in i.metadata.name:
        ip_lists.append(i.status.pod_ip)
        logger.info("Pod IP {}".format(i.status.pod_ip))
        # 직접 container 하고 통신 실험 진행 할 때
        # temp = "192.9.85.158"
        # temp = "172.17.0.2"
        # i.status.pod_ip = temp
        # clients = []
        # print("nnn")
        # url = "http://" + i.status.pod_ip + ':8585/'
        # fl_server = FL_server(url, 1)
        # fl_server.start("initial")

print(ip_lists)

# Jeston 하고 통신 할 객체 생성
# TODO: 현재는 두대로 고정, Num_users로 받아서 개수에 따라 하도록 조정
fl_clients = [FL_server(), FL_server()]
fl_server = FL_server()

num_users = fl_server.args["num_users"]

# Synchronize 통신
pool = ThreadPoolExecutor(max_workers=10)
jobs = []

accs = []
losses = []
train_losses = []
test_dataset = get_dataset(fl_server.args["dataset"])


# Initialize (
for i in range(num_users):
    print(i)
    jobs.append(pool.submit(fl_clients[i].initialize, i, ip_lists[i]))


if wait_finish(jobs) == "Finish":
    pass

# Training
for r in range(fl_server.args["epochs"]):
    jobs = []
    for i in range(num_users):
        jobs.append(pool.submit(fl_clients[i].start, fl_server.global_model))

    if wait_finish(jobs) == "Finish":
        fl_server.global_model = aggregate(fl_server.global_model, num_users)

    temp = [float(fl_clients[i].train_loss) for i in range(num_users)]
    train_losses.append(sum(temp) / len(temp))

    acc, loss = test_inference(fl_server.global_model, test_dataset)

    accs.append(acc)
    losses.append(loss)

    print("Accuracy after {}".format(r), acc)


write_text_file(accs, losses, train_losses)

