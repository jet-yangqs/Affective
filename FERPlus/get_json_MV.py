# coding=utf-8

import csv
import json
from PIL import Image
import os
import numpy as np

def floateq(x, y):
    return abs(x - y) < 1e-7


directory = 'database/Training'
cropsize = 42
imsize = 48
t = (imsize - cropsize) / 2
lentag = 8

filename = 'fer2013new.csv'
d = dict()
jsond = dict()
with open(filename) as f:
    reader = list(csv.reader(f))

for cnt, line in enumerate(reader):
    if cnt == 0:
        continue
    line0 = line[0].strip()
    line1 = line[1].strip()
    if line1 == '':
        continue
    print('database/' + line0 + '/' + line1)
    temp = list(map(lambda x: float(x)/(10.0-float(line[-2])), line[2:-2]))
    if abs(sum(temp) - 1.0) >= 1e-7:
          continue
    maxtemp = max(temp)
    if maxtemp <= 0.5:
       continue
    temp2 = []
    for element in temp:
          if element == maxtemp:
               temp2.append(1.0)
          else:
               temp2.append(0.0)
    d['database/' + line0 + '/' + line1] = temp2
    assert abs(sum(temp2) - 1.0) < 1e-7
 

with open('data.json', 'w') as f:
    json.dump(d, f)


jsond['Training'] = list()
pics = os.listdir(directory)
for pic in pics:
    path = directory + '/' + pic
    print('Train', path)
    if path not in d.keys():
        continue
    image = Image.open(path)
    a1 = image.crop([0, 0, cropsize, cropsize])
    b1 = image.crop([0, imsize-cropsize, cropsize, imsize])
    c1 = image.crop([imsize-cropsize, 0, imsize, cropsize])
    d1 = image.crop([imsize-cropsize, imsize-cropsize, imsize, imsize])
    e1 = image.crop([t, t, imsize-t, imsize-t])
    a2 = a1.transpose(Image.FLIP_LEFT_RIGHT)
    b2 = b1.transpose(Image.FLIP_LEFT_RIGHT)
    c2 = c1.transpose(Image.FLIP_LEFT_RIGHT)
    d2 = d1.transpose(Image.FLIP_LEFT_RIGHT)
    e2 = e1.transpose(Image.FLIP_LEFT_RIGHT)
    l = [a1, b1, c1, d1, e1, a2, b2, c2, d2, e2]
    e = list(map(lambda x: np.array(x), l))
    for element in e:
        tup = (element.tolist(), d[path], path)
        jsond['Training'].append(tup)


tests = ['database/PublicTest', 'database/PrivateTest']
m = {'database/PublicTest':'PublicTest', 'database/PrivateTest':'PrivateTest'}
for test in tests:
    if test not in jsond.keys():
        jsond[m[test]] = list()
    pics = os.listdir(test)
    for pic in pics:
        path = test + '/' + pic
        if path not in d.keys():
           continue
        print(path)
        image = Image.open(path)
        a1 = image.crop([0, 0, cropsize, cropsize])
        b1 = image.crop([0, imsize - cropsize, cropsize, imsize])
        c1 = image.crop([imsize - cropsize, 0, imsize, cropsize])
        d1 = image.crop([imsize - cropsize, imsize - cropsize, imsize, imsize])
        e1 = image.crop([t, t, imsize - t, imsize - t])
        a2 = a1.transpose(Image.FLIP_LEFT_RIGHT)
        b2 = b1.transpose(Image.FLIP_LEFT_RIGHT)
        c2 = c1.transpose(Image.FLIP_LEFT_RIGHT)
        d2 = d1.transpose(Image.FLIP_LEFT_RIGHT)
        e2 = e1.transpose(Image.FLIP_LEFT_RIGHT)
        l = [a1, b1, c1, d1, e1, a2, b2, c2, d2, e2]
        e = list(map(lambda x: np.expand_dims(np.array(x), axis=0), l))
        concat_e = np.concatenate(e, axis=0)
        tag = d[path]
        tags = [np.expand_dims(tag, axis=0) for dup in range(10)]
        taglist = np.concatenate(tags, axis=0)
        jsond[m[test]].append((concat_e.tolist(), taglist.tolist(), path+ '/' +pic))

print(jsond.keys())
with open("info_MV.json", "w") as f:
    json.dump(jsond, f)
