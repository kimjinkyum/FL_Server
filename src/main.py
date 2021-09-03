from kubernetes import client, config
import requests

from options import *
from FL_server import FL_server
from concurrent.futures import ThreadPoolExecutor
from utils import *

config.load_kube_config()
v1 = client.CoreV1Api()

print("Listing pods with their IPs")

ret = v1.list_pod_for_all_namespaces(watch=False)
ip_lists = []
for i in ret.items:
    # print("%s\t%s\t%s" % (i.status.pod_ip, i.metadata.namespace, i.metadata.name))
    pass
    if "fl-ex" in i.metadata.name :
        headers = {}
        ip_lists.append(i.status.pod_ip)
        # ip_lists.append(i.status.pod_ip)

        #temp = "192.9.85.158"
        # temp = "172.17.0.2"
        #i.status.pod_ip = temp
        # clients = []
        # print("nnn")
        # url = "http://" + i.status.pod_ip + ':8585/'
        # fl_server = FL_server(url, 1)
        # fl_server.start("initial")
print(ip_lists)

fl_clients = [FL_server(), FL_server()]
fl_server = FL_server()

num_users = fl_server.args["num_users"]
pool = ThreadPoolExecutor(max_workers=10)
jobs = []

accs = []
losses = []
train_losses = []
test_dataset = get_dataset(fl_server.args["dataset"])
# Initialize
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

