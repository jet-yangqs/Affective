# coding=utf-8

import cv2
import os
import numpy as np
import re
from copy import deepcopy
import csv
import json
mapping = {'happiness': 0, 'others': 1, 'repression': 2, 'surprise': 3, 'disgust': 4}

def readcsv(file='CASME2-coding-20140508.csv'):
    with open(file) as f:
        reader = list(csv.reader(f))
        info = dict()
        for cnt, element in enumerate(reader):
            # If the number in excel is 1,then modify it to 01
            realdir = element[0] if len(element[0]) == 2 else '0' + element[0]
            sentiment = element[8].strip()
            if sentiment in mapping.keys():
                info['sub' + realdir + '/' + element[1]] = mapping[sentiment]
    return info


info = readcsv()
dir = 'Cropped'
files = os.listdir(dir)
threshold1 = 80
threshold2 = 5
size = 120
edge = 4
info2 = dict()
for file in files:
    layer2 = dir + '/' + file
    files2 = os.listdir(layer2)
    print(files2)
    for file2 in files2:
        layer3 = layer2 + '/' + file2
        if layer3.find('_Optical') >= 0:
            continue
        t = layer3 + '_Optical'
        if not os.path.exists(t):
            os.mkdir(t)
        frames = os.listdir(layer3)
        frames = list(map(lambda x:(x, int(re.findall('(?:)(\d+)(?:)', x)[0])), frames))
        frames = sorted(frames, key=lambda x: x[1])
        start = frames[0][1]
        length = frames[-1][1]
        print(file2, start, length)
        cnt = start
        image = cv2.imread('%s/reg_img%s.jpg' % (layer3, cnt))
        prvs = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        prvs = cv2.resize(prvs, (size, size))
        optical_flow = cv2.DualTVL1OpticalFlow_create()
        hsv = np.zeros((size, size, 3))
        hsv = hsv.astype(np.uint8)
        hsv[..., 1] = 255
        cnt += 1
        name = layer3.replace('Cropped/', '')
        if name not in info.keys():
            print('%s not in info.keys()!')
            continue
        max_amplitude = 0.0
        A = dict()
        while cnt <= length and name in info.keys():
            print('%s/reg_img%s.jpg'%(layer3, cnt))
            newimage = cv2.imread('%s/reg_img%s.jpg'%(layer3, cnt))
            next = cv2.cvtColor(newimage, cv2.COLOR_BGR2GRAY)
            next = cv2.resize(next, (size, size))
            flow = optical_flow.calc(prvs, next, None)
            mag, ang = cv2.cartToPolar(flow[..., 0], flow[..., 1])
            hsv[..., 0] = ang * 180 / np.pi / 2
            hsv[..., 2] = cv2.normalize(mag, None, 0, 255, cv2.NORM_MINMAX)
            bgr = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
            newimage = deepcopy(bgr)
            newimage[:, :, 0] = np.maximum(bgr[:, :, 0] - np.mean(bgr[:, :, 0]), 0)
            newimage[:, :, 1] = np.maximum(bgr[:, :, 1] - np.mean(bgr[:, :, 1]), 0)
            newimage[:, :, 2] = np.maximum(bgr[:, :, 2] - np.mean(bgr[:, :, 2]), 0)
            newimage[np.where(newimage < threshold1)] = 0
            newimage = newimage[edge:size-edge, edge:size-edge]
            cv2.imwrite('%s/frame%s.bmp' % (t, cnt), newimage)
            next = prvs
            sum = np.sum(np.square(newimage))
            if sum > max_amplitude:
                max_amplitude = sum
            A['%s/frame%s.bmp' % (t, cnt)] = sum
            cnt += 1
        for key in A.keys():
            if A[key] > 1e-7 and max_amplitude/A[key] < threshold2:
                info2[key] = info[name]
                print('record %s,sum=%s' % (key, A[key]))

with open('info2.json', 'w') as f:
    json.dump(info2, f)
