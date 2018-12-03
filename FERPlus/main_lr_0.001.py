# coding=utf-8

import tensorflow as tf
import numpy as np
import json
import random
import traceback
import time
from copy import deepcopy

batchsize = 100
filters1 = 44
filters2 = 44
filters3 = 88
size = 42
middle1 = 2048
middle2 = 1024
taglength = 8
init = 0.001
modelstep = 800
decay = 0.995
batchcursor = dict()
pathtomodel = "main_lr_0.001/"
maxoverflow = 5000
maxacc1 = 0.0
argmaxacc1 = 0
maxacc2 = 0.0
argmaxacc2 = 0
ISOTIMEFORMAT = '%Y-%m-%d %H:%M:%S'
BN_EPSILON = 0.0001
bnname = 0
drop_out_rate=0.6
taskfirst=False
confusion = dict()
for cnt1 in range(taglength):
    for cnt2 in range(taglength):
        confusion["%s,%s"%(cnt1,cnt2)] = 0
def prelu(x, alpha=0.01):
    return tf.maximum(alpha*x,x)  
 
def shuffle():
    for key in d.keys():
        random.shuffle(d[key])
        batchcursor[key] = 0


def generate_shape_tuple(pre):
    shape = pre.shape
    li = []
    li.append(1)
    for element in shape:
        li.append(element)
    finaltype = tuple(li)
    return deepcopy(pre.reshape(finaltype))


def getbatch(type, batchsize):
    l = d.get(type, False)
    print(d.keys())
    if not l or not isinstance(batchsize, int):
        return None
    cnt = 0
    start = batchcursor[type]
    ret_data = []
    ret_tag = []
    length = len(l)
    overflow = False
    while cnt < batchsize:
        npdata = np.array(l[start][0])
        nptag = np.array(l[start][1])
        npdatareshape = generate_shape_tuple(npdata)
        nptagreshape = generate_shape_tuple(nptag)
        ret_data.append(npdatareshape)
        ret_tag.append(nptagreshape)
        cnt += 1
        pre = start
        start = (start + 1) % length
        batchcursor[type] = (batchcursor[type] + 1) % length
        if start < pre:
            overflow = True
            if type.find('Test') >= 0:
                break
    m = np.concatenate(ret_data, axis=0)
    n = np.concatenate(ret_tag, axis=0)
    return m.astype(np.float32), n.astype(np.float32), overflow


def get_feeddict(type, batchsize):
    m, n, overflow = getbatch(type, batchsize)
    #print(m)
    if type.find('Training') >= 0:
        return {keep_prob: drop_out_rate, x: m, y: n}, overflow
    else:
        return {keep_prob: 1.0, x: m, y: n}, overflow


def clear_cursor():
    for key in batchcursor.keys():
        batchcursor[key] = 0


def handle_with_test(task):
    if task not in ['PublicTest', 'PrivateTest']:
       return
    for key in d.keys():
        print(len(d[key]))
    l = d[task]
    correct = 0
    for cnt, element in enumerate(l):
       fd = {x:element[0], y:element[1], keep_prob: 1.0}
       predictions = sess.run(softmax, feed_dict=fd)
       m = np.zeros(taglength)
       for prediction in predictions:
           m += prediction
       predict_value = np.argmax(m)
       true_value = np.argmax(element[1][0])
       confusion["%s,%s"%(true_value,predict_value)] += 1
       if predict_value == true_value:
            correct += 1
       else:
            print("[task=%s, image=%s]Predict=%s, Actual=%s"%(task,element[2], predict_value, true_value))
       print("Testing %s...cnt=%s,correct=%s, length=%s"%(task, cnt,correct, len(d[task])))
    acc = float(correct)/len(l)
    print("correct=%s, len(d[task])=%s, acc=%s"%(correct, len(d[task]), acc))
    for cnt1 in range(taglength):
        for cnt2 in range(taglength):
            print("actual=%s,predict=%s,num=%s"%(cnt1,cnt2,confusion["%s,%s"%(cnt1,cnt2)]))
    with open("confusion.json","w") as f:
         json.dump(confusion,f)
    for cnt1 in range(taglength):
        for cnt2 in range(taglength):
            confusion["%s,%s"%(cnt1,cnt2)] = 0
    return acc


def do_task(maxoverflow, task, clearcursor=True, epochno=0):
    print("Current task is %s"%task)
    overflow = False
    cnt = 0
    count = 0
    count_rights = 0
    if clearcursor:
        clear_cursor()
    acc = 0.0
    while cnt < maxoverflow:
        t1 = time.time()
        feeddict, overflow = get_feeddict(task, batchsize)
        going_to_exit = False
        feeddict[keep_prob] = 1.0
        if overflow:
             if task.find("Training") >= 0:
                 saver.save(sess, "%s/hello"%pathtomodel, global_step=globalstep)
                 print("save the model")
                 shuffle()
                 cnt += 1
        n1, n2, n3,learning_rate, h_fc1_drop_data, gradvalue = sess.run([accuracy, loss, globalstep,lr,result, m5],feed_dict=feeddict)
        length_batch = len(h_fc1_drop_data)
        count += 1
        feeddict[keep_prob] = drop_out_rate
        if task.find("Training") >= 0:
             sess.run(train_step, feed_dict=feeddict)  
        print(n1, n2, "epoch=%s, overflow=%s, count=%s, globalstep=%s, time used=%s, lr=%s"%(epochno, cnt, count, n3, time.time()-t1, learning_rate))
        print("epoch=%s, maxacc1=%s,argmaxacc1=%s(PrivateTest).maxacc2=%s,argmaxacc2=%s(PublicTest)"%(epochno, maxacc1, argmaxacc1, maxacc2, argmaxacc2))
        print(batchcursor)
        #print(gradvalue, np.max(gradvalue))
    return n1


print("Data loading")
with open("info_DIS.json") as f:
    d = json.load(f)
shuffle()
lengthpublic = len(d["PublicTest"])


x = tf.placeholder(tf.float32, [None, size, size])
y = tf.placeholder(tf.float32, [None, taglength])
keep_prob = tf.placeholder(tf.float32)
globalstep = tf.Variable(0, dtype=tf.int32, trainable=False)


def flatten(x):
    shape = x.get_shape()
    mul = 1
    lengthshape = len(shape)
    if lengthshape == 1:
        return x
    for cnt in range(1, lengthshape):
        mul *= int(shape[cnt])
    return tf.reshape(x, [-1, mul])


def relu(x):
    return tf.nn.relu(x)


def dropout(x):
    return tf.nn.dropout(x, keep_prob)


def expand(x, dim=-1):
    return tf.expand_dims(x, dim)


def conv(x, filternum, strides=1, padding="same", kernelsize=2, isdropout=False, expand=False, bias=0.1):
    if expand:
        x = tf.expand_dims(x, -1)
    x1 = tf.layers.conv2d(x, filternum, kernel_size=kernelsize, strides=strides, padding=padding
                          , data_format="channels_last", bias_initializer=tf.constant_initializer(bias))
    if isdropout:
        x1 = dropout(x1)
    return x1


def batchnorm(x):
    global bnname
    betaname = 'beta%s' % bnname
    gammaname = 'gamma%s' % bnname
    shape = x.get_shape()
    axeslist = [cnt for cnt in range(len(shape) - 1)]
    dimension = shape.as_list()[-1]
    mean, variance = tf.nn.moments(x, axes=axeslist)
    beta = tf.get_variable(betaname, dimension, tf.float32, initializer=tf.constant_initializer(1, tf.float32))
    gamma = tf.get_variable(gammaname, dimension, tf.float32, initializer=tf.constant_initializer(1, tf.float32))
    x1 = tf.nn.batch_normalization(x, mean, variance, beta, gamma, BN_EPSILON)
    bnname += 1
    return x1


def resblock(x, filternum, kernelsize=3, isdropout=False, isopen=True):
    if not isopen:
        return x
    kernels = []
    if isinstance(kernelsize,  int):
        kernels.extend([kernelsize, kernelsize])
    elif (isinstance(kernelsize, tuple) or isinstance(kernelsize, list)) and len(kernelsize) == 2:
        kernels = list(kernelsize)
    else:
        raise ValueError("The shape of kernelsize is wrong.It must be a integer or a tuple/list with 2 elements")
    x1 = conv(x, filternum, kernelsize=kernels[0], isdropout=isdropout)
    x2 = batchnorm(x1)
    x3 = tf.nn.relu(x2)
    x4 = conv(x3, filternum, kernelsize=kernels[1], isdropout=isdropout)
    x5 = batchnorm(x4)
    x6 = tf.add(x, x5)
    return relu(x6)


def pooling(x, pool_size=2, strides=2, method='max', padding='valid'):
    if method == 'max':
        x1 = tf.layers.max_pooling2d(x, pool_size=pool_size, strides=strides, padding=padding)
    elif method == 'average':
        x1 = tf.layers.average_pooling2d(x, pool_size=pool_size, strides=strides, padding=padding)
    else:
        x1 = x
    return x1


def microdense(x, filternum, kernelsize=3):
    x1 = batchnorm(x)
    x2 = relu(x1)
    x3 = conv(x2, filternum, kernelsize=kernelsize, isdropout=True)
    return x3


def denseblock(x, filternum, kernelsize=3, densenum=3):
    tensors = [x]
    for cnt in range(densenum):
        sumtensors = tf.add_n(tensors)
        y = microdense(sumtensors, filternum, kernelsize)
        tensors.append(y)
    return tf.add_n(tensors)


def minidensenet(x, filternum, kernelsize=3, densenum=3, blocks=2, res=True):
    cur = x
    shape = cur.get_shape()[-1]
    for cnt in range(blocks):
        newtensor = denseblock(cur, filternum, kernelsize, densenum)
        cur = newtensor
    if res:
        return cur + x
    else:
        return cur


print("x.get_shape()", x,"Note that x is a tensorflow placeholder!(input)")
m1 = expand(x)
print("m1.get_shape()", m1.get_shape())
m2 = prelu(conv(m1, filters1, strides=1, kernelsize=5, padding="valid"))
print("m2.get_shape()", m2.get_shape())
m3 = tf.pad(m2, [[0, 0], [2, 2], [2, 2],[0,0]])
print("m3.get_shape()", m3.get_shape(), "Written in the paper(convolution1)")
m4 = pooling(m3, pool_size=2, strides=2, padding='same')
print("m4.get_shape()", m4.get_shape(), "Written in the paper(pooling1)")
m5 = prelu(conv(m4, filters2, strides=1, kernelsize=3, padding="valid"))
print("m5.get_shape()", m5.get_shape())
m6 = tf.pad(m5, [[0, 0], [1, 1], [1, 1],[0,0]])
print("m6.get_shape()", m6.get_shape(), "Written in the paper(convolution2)")
m7 = pooling(m6, pool_size=2, strides=2, padding='same')
print("m7.get_shape()", m7.get_shape(), "Written in the paper(pooling2)")
m8 = prelu(conv(m7, filters3, strides=1, kernelsize=5, padding="valid"))
print("m8.get_shape()", m8.get_shape())
m9 = tf.pad(m8, [[0, 0], [2, 2], [2, 2],[0, 0]])
print("m9.get_shape()", m9.get_shape(), "Written in the paper(convolution3)")
m10 = pooling(m9, pool_size=2, strides=2, padding='same')
print("m10.get_shape()", m10.get_shape(), "Written in the paper(pooling3)")
m11 = flatten(m10)
print("m11.get_shape()", m11.get_shape()) 
m12 = prelu(tf.layers.dense(m11, middle1, activation=None, kernel_initializer=tf.truncated_normal_initializer(stddev=0.1, seed=1),bias_initializer=tf.constant_initializer(0.1)))
print("m12.get_shape()", m12.get_shape())
m13 = dropout(m12)
print("m13=dropout(m12), get_shape()=", m13.get_shape(), "Written in the paper(FC1)")
m14 = prelu(tf.layers.dense(m13, middle2, activation=None, kernel_initializer=tf.truncated_normal_initializer(stddev=0.1, seed=1),bias_initializer=tf.constant_initializer(0.1)))
print("m14.get_shape()",m14.get_shape())
m15 = dropout(m14)
print("m15=dropout(m14), m15.get_shape()", m15.get_shape(), "Written in the paper(FC2)")
result = tf.layers.dense(m15, taglength, activation=None, kernel_initializer=tf.truncated_normal_initializer(stddev=0.1, seed=1),bias_initializer=tf.constant_initializer(0.1))
print("result.get_shape()",result.get_shape(), "Written in the paper,output of FC3.Note that please do not use relu in the last layer!")
softmax = tf.nn.softmax(result)
print("softmax-->cross entropy loss...")

loss = tf.reduce_mean(
    tf.nn.softmax_cross_entropy_with_logits(labels=y, logits=result))
grad = tf.gradients(loss, m11)
lr = tf.train.exponential_decay(init, globalstep, modelstep, decay, staircase=False)
original_opt = tf.train.AdamOptimizer(lr)
opt = tf.contrib.estimator.clip_gradients_by_norm(original_opt, clip_norm=4.0)
train_step = opt.minimize(loss, global_step=globalstep)
correct_prediction = tf.equal(tf.argmax(result, 1), tf.argmax(y, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))


saver = tf.train.Saver()
for var in tf.global_variables():
    print(var.op.name)
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    try:
        saver.restore(sess, tf.train.latest_checkpoint(pathtomodel))
        print("Restore data successfully!")
    except:
        print("Error when reading model")
        print(traceback.format_exc())
        pass
    temp = 1
    record = dict()
    for epoch in range(maxoverflow):
        if taskfirst:
            n1=do_task(temp, "Training", False, epoch)
            acc1 = handle_with_test("PrivateTest")
            acc2 = handle_with_test("PublicTest")
        else:
            acc1 = handle_with_test("PrivateTest")
            acc2 = handle_with_test("PublicTest")
            n1=do_task(temp, "Training", False,epoch)
        record[epoch] = {"acc1":str(acc1), "acc2":str(acc2),"n1":str(n1)}
        with open("record.json","w") as f:
            json.dump(record,f)
        if acc1 > maxacc1:
              maxacc1 = acc1
              argmaxacc1 = epoch
        if acc2 > maxacc2: 
              maxacc2 = acc2
              argmaxacc2 = epoch
        print("\n[time=%s, epoch=%s]accPrivate=%s,accPublic=%s,maxaccPrivate=%s,maxaccPublic=%s,argmaxaccPrivate=%s,argmaxaccPublic=%s\n" % (time.strftime(ISOTIMEFORMAT, time.localtime()), epoch, acc1, acc2, maxacc1, maxacc2, argmaxacc1, argmaxacc2))
        print(record)




