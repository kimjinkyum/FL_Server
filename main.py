from kubernetes import client, config
import requests
from options import *
from FL_server import FL_server
from concurrent.futures import ThreadPoolExecutor, as_completed

config.load_kube_config()
v1 = client.CoreV1Api()

print("Listing pods with their IPs")

ret = v1.list_pod_for_all_namespaces(watch=False)
for i in ret.items:
    # print("%s\t%s\t%s" % (i.status.pod_ip, i.metadata.namespace, i.metadata.name))
    pass
    if i.metadata.name == "fl-ex":
        headers = {}

        temp = "192.9.85.158"
        i.status.pod_ip = temp
        clients = []
        url = "http://" + i.status.pod_ip + ':8585/'
        fl_server = FL_server(url)
        if i.metadata.name == "fl-ex":
            with ThreadPoolExecutor(max_workers=3) as pool:
                jobs = [pool.submit(fl_server.start, *vars) for vars in zip ([1, 2] , ["initial", "initial"])]

                for job in as_completed(jobs):
                    print(job.result())


