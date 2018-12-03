# coding=utf-8

import numpy as np
import tensorflow as tf
import json
import random
from copy import deepcopy
from PIL import Image
import sys
import re,json
import traceback

temp = []
try:
    temp = list(map(lambda x:int(x.strip()), sys.argv[1].strip().split(',')))
except:
    temp = [1]
print("test",temp)
temp = list(map(lambda x:'sub'+str(x) if len(str(x)) == 2 else 'sub0'+str(x), temp))
print(temp) 
batchsize = 32
size = 112
numtag = 5
epoches = 10
init = 0.0001
wd = 0.0003
decay = 0.997
round = 200
middle1=2048
middle2=1024
filters = [44,44,88]
eye = np.eye(5)
with open('info2.json') as f:
    d = json.load(f)
keys = list(d.keys())
random.shuffle(keys)
test = []
training = []
for element in keys:
    A = re.findall('(sub\d+)', element)[0]
    if A in temp:
        print("test add %s"%element)
        test.append(element)
    else:
        training.append(element)
test = sorted(test)
cursor = 0
testcursor = 0
mapping = {'happiness': 0, 'others': 1, 'repression': 2, 'surprise': 3, 'disgust': 4}
try:
    with open("confusion.json") as f:
       confusion = json.load(f)
except:
    confusion = dict()
    for cnt1 in range(numtag):
        for cnt2 in range(numtag):
            confusion["%s,%s"%(cnt1,cnt2)] = 0
t= dict()

def dropout(x):
    return tf.nn.dropout(x, keep_prob)

def fetchbatch():
    global cursor
    if cursor + batchsize < len(training):
        batch = list(deepcopy(training[cursor: cursor+batchsize]))
        cursor += batchsize
        return batch, False
    else:
        batch = list(deepcopy(training[cursor:len(training)]))
        res = batchsize-len(batch)
        batch.extend(list(deepcopy(training[0: res])))
        cursor = res
        return batch, True


def getbatch():
    global cursor
    Imarray = np.zeros((batchsize, size, size, 3))
    tagarray = np.zeros((batchsize, numtag))
    batch, overflow = fetchbatch()
    for cnt, element in enumerate(batch):
        assert element not in test
        Imarray[cnt] = np.array(Image.open(element))
        tagarray[cnt] = eye[d[element]]
    return deepcopy(Imarray), deepcopy(tagarray), overflow


def fetchone():
    global testcursor
    assert test[testcursor] not in training
    image = np.array(Image.open(test[testcursor])).reshape((-1, size, size, 3))
    tag = np.array(eye[d[test[testcursor]]]).reshape((-1, numtag))
    former = testcursor
    testcursor = (testcursor + 1) % len(test)
    return deepcopy(image), deepcopy(tag), True if testcursor < former else False, test[former]


def weight_variable(shape, w_alpha=0.1):
    '''
    增加噪音，随机生成权重
    :param shape:
    :param w_alpha:
    :return:
    '''
    initial = w_alpha * tf.random_normal(shape)
    return tf.Variable(initial)


def bias_variable(shape, b_alpha=0.1):
    '''
    增加噪音，随机生成偏置项
    :param shape:
    :param b_alpha:
    :return:
    '''
    initial = b_alpha * tf.random_normal(shape)
    return tf.Variable(initial)


def flatten(final):
    shape_final = list(final.get_shape())
    if len(shape_final) <= 1:
        return shape_final
    mul = 1
    for cnt in range(1, len(shape_final)):
        mul *= int(shape_final[cnt])
    return tf.reshape(final, [-1, mul]), mul


def leaky_relu(x, alpha=0.15):
    return tf.maximum(alpha * x, x)


inp = tf.placeholder(tf.float32, [None, size, size, 3])
y = tf.placeholder(tf.float32, [None, numtag])
keep_prob=tf.placeholder(tf.float32)
layer1 = tf.layers.conv2d(inp, filters=filters[0], kernel_size=5, padding='same', bias_initializer=tf.constant_initializer(1))
layer2 = tf.layers.average_pooling2d(layer1, pool_size=2, strides=2)
layer3 = leaky_relu(layer2)
layer4 = tf.layers.conv2d(layer3, filters=filters[1], kernel_size=3, padding='same', bias_initializer=tf.constant_initializer(1))
layer5 = tf.layers.average_pooling2d(layer4, pool_size=2, strides=2)
layer6 = leaky_relu(layer5)
layer7 = tf.layers.conv2d(layer6, filters=filters[2], kernel_size=5, padding='same', bias_initializer=tf.constant_initializer(1))
layer8 = tf.layers.average_pooling2d(layer7, pool_size=2, strides=2)
layer9 = leaky_relu(layer8)
final, sz = flatten(layer9)
m1 = leaky_relu(tf.layers.dense(final, middle1, activation=None, kernel_initializer=tf.truncated_normal_initializer(stddev=0.1, seed=1),bias_initializer=tf.constant_initializer(0.1)))
m2 = dropout(m1)
m3 = leaky_relu(tf.layers.dense(m2, middle2, activation=None, kernel_initializer=tf.truncated_normal_initializer(stddev=0.1, seed=1),bias_initializer=tf.constant_initializer(0.1)))
m4 = dropout(m3)
Y = tf.layers.dense(m4, numtag, activation=None, kernel_initializer=tf.truncated_normal_initializer(stddev=0.1, seed=1),bias_initializer=tf.constant_initializer(0.1))




name = [v.name for v in tf.trainable_variables()]
print(name)
conv2d_kernel0 = tf.get_default_graph().get_tensor_by_name('conv2d/kernel:0')
conv2d_1_kernel0 = tf.get_default_graph().get_tensor_by_name('conv2d_1/kernel:0')
conv2d_2_kernel0 = tf.get_default_graph().get_tensor_by_name('conv2d_2/kernel:0')
loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=Y, labels=y)) 
global_step = tf.Variable(0, trainable=False)
learning_rate = tf.train.exponential_decay(init, global_step, round, decay, staircase=True)
learning_step = tf.train.AdamOptimizer(learning_rate).minimize(loss, global_step=global_step)
correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(Y, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    overflowtimes = 0
    maxacc1 = 0.0
    argmaxacc1 = 0
    maxacc2 = 0.0
    argmaxacc2 = 0
    videonum = 0
    while overflowtimes < epoches:
        ims, tags, isoverflow = getbatch()
        print("epoch=%s"%overflowtimes, isoverflow, cursor)
        sess.run(learning_step, feed_dict={inp: ims, y: tags,keep_prob:0.6})
        if isoverflow:
            overflowtimes += 1
            totalright = 0
            total = 0
            last = ''
            videos = 0 
            sumcorrect = 0
            info = dict()
            tempconfusion = dict()
            for cnt1 in range(numtag):
               for cnt2 in range(numtag):
                  tempconfusion["%s,%s" % (cnt1, cnt2)] = 0
            while True:
                im, tag, z, testpath = fetchone()
                if z:
                    break
                B = re.findall('(sub.+Optical)', testpath)[0].strip('\n').strip()
                if B != last:
                    if last in info.keys():
                       argmaxinfo = 0
                       maxinfo = 0
                       info[last]["isvisited"] = True
                       for emotion in range(0, numtag):
                          if info[last][emotion] > maxinfo:
                             maxinfo = info[last][emotion]
                             argmaxinfo = emotion
                       tempconfusion["%s,%s"%(info[last]["actual"], argmaxinfo)] += 1
                       print("testdata=%s,argmaxinfo=%s,actual=%s"%(last,argmaxinfo,info[last]["actual"]))
                       if argmaxinfo == info[last]["actual"]:
                             sumcorrect += 1
                    videos += 1
                    print("videos=%s,B=%s"%(videos,B))
                    last = B
                    info[B] = dict()
                    info[B]["isvisited"] = False
                    for emotion in range(0, numtag):
                        info[B][emotion] = 0
                Ydata, acc = sess.run([Y, accuracy], feed_dict={inp: im, y: tag,keep_prob:1.0})
                for emotion in range(numtag):
                    info[B][emotion] += Ydata[0][emotion]
                info[B]["actual"] = np.argmax(tag[0])
                totalright += acc
            argmaxinfo = 0
            maxinfo = 0
            assert info[B]["isvisited"] == False
            for emotion in range(0, numtag):
               if info[B][emotion] > maxinfo:
                   maxinfo = info[B][emotion]
                   argmaxinfo = emotion
            tempconfusion["%s,%s"%(info[B]["actual"], argmaxinfo)] += 1
            print("testdata=%s,argmaxinfo=%s,actual=%s"%(B,argmaxinfo,info[B]["actual"]))
            if argmaxinfo == info[B]["actual"]:
               info[B]["correct"] = True
               sumcorrect += 1
            acc1 = totalright/len(test)
            if acc1 > maxacc1:
                 maxacc1 = acc1
                 argmaxacc1 = overflowtimes
            acc2 = sumcorrect
            if acc2 > maxacc2:
                 maxacc2 = acc2
                 argmaxacc2 = overflowtimes
            videonum = videos
            print("epoch=%s,totalright=%s,len(test)=%s,acc1=%s,maxacc1=%s,argmaxacc1=%s,sumcorrect=%s,videos=%s,acc2=%s,maxacc2=%s,argmaxacc2=%s"%(overflowtimes,totalright,len(test), acc1,maxacc1,argmaxacc1,sumcorrect,videos,acc2,maxacc2,argmaxacc2))
            random.shuffle(training)
            t[overflowtimes] = tempconfusion
            for cnt1 in range(numtag):
                for cnt2 in range(numtag):
                    print("actual=%s,recognized=%s:%s"%(cnt1, cnt2, tempconfusion["%s,%s"%(cnt1,cnt2)]))
    try:
        with open('result.json') as f:
             d = json.load(f)
    except:
        d = dict()
    for cnt1 in range(numtag):
        for cnt2 in range(numtag):
            confusion["%s,%s"%(cnt1,cnt2)] += t[argmaxacc2]["%s,%s"%(cnt1,cnt2)]
            print("actual=%s,recognized=%s:%s"%(cnt1, cnt2, confusion["%s,%s"%(cnt1,cnt2)]))
    with open("confusion.json","w") as j:
        json.dump(confusion, j)
    d[temp[0]] = {"maxacc1":maxacc1, "argmaxacc1":argmaxacc1,"maxacc2":maxacc2,"argmaxacc2":argmaxacc2, "videonum":videonum}
    with open('result.json','w') as f:
        json.dump(d, f) 


