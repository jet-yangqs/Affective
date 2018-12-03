#  coding=utf-8

import os,json
import time
import logging
logging.basicConfig(filename='execute.log',
                    format='%(asctime)s -%(name)s-%(levelname)s-%(module)s:%(message)s',
                    datefmt='%Y-%m-%d %H:%M:%S %p',
                    level=logging.DEBUG)
classes = 27
if os.path.exists('confusion.json'):
    with open('confusion.json','w') as f:
        pass
for cnt in range(1, classes):
    logging.info('python3 recog_SHCNN.py %s'%cnt)
    os.system("python3 recog_SHCNN.py %s"%cnt)

with open('result.json') as f:
    d = json.load(f)
print(d)
s = 0.0
for key in d.keys():
    s += float(d[key]["maxacc2"])
print(s/246)
