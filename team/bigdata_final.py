from __future__ import print_function
import tensorflow as tf
import numpy as np
import csv
import pandas as pd

def read_file(file_name):
    if(file_name=="train.csv"):
        fields = ['full_sq','railroad_station_walk_km', 'public_transport_station_km','price_doc']
        df = pd.read_csv('train.csv', skipinitialspace=True, usecols=fields)
        # shape:(x,?) => (x,1)
        x1 = df[fields[0]].values # to nummpy array
        x1 = x1.reshape(len(x1),1)
        x2 = df[fields[1]].values
        x2 = x2.reshape(len(x2),1)
        x3 = df[fields[2]].values
        x3 = x3.reshape(len(x3),1)
        y = df[fields[3]].values
        y = y.reshape(len(y),1)
        return x1,x2,x3,y
    elif(file_name=="test.csv"):
        fields = ['id','full_sq','railroad_station_walk_km', 'public_transport_station_km']
        df = pd.read_csv('test.csv', skipinitialspace=True, usecols=fields)
        # shape:(x,?) => (x,1)
        ID = df[fields[0]].values
        ID = ID.reshape(len(ID),1)
        x1 = df[fields[1]].values
        x1 = x1.reshape(len(x1),1)
        x2 = df[fields[2]].values
        x2 = x2.reshape(len(x2),1)
        x3 = df[fields[3]].values
        x3 = x3.reshape(len(x3),1)
        return ID,x1,x2,x3

def write_file(col1, col2):
    with open('result.csv','w') as f:
        writer = csv.writer(f)
        writer.writerow(['id','price_doc'])
        for i in range(len(col1)):
            t = int(col2[i][0])
            t = np.absolute(t)
            writer.writerow([col1[i][0],t])

def add_layer(input1, input2, input3, in_size, out_size, n_layer,activation_function=None):
    layer_name = 'layer%s' % n_layer
    with tf.name_scope(layer_name):
        if input2 is None:
            w = tf.Variable(tf.random_normal([in_size, out_size]), name='w')
            b = tf.Variable(tf.zeros([1,out_size]) + 0.1, name='b')
            Wx_plus_b = tf.matmul(input1, w) + b
            if activation_function is None:
                outputs = Wx_plus_b
            else:
                outputs = activation_function(Wx_plus_b)
            tf.summary.histogram('/output', outputs)
            return outputs
        else:
            w1 = tf.Variable(tf.random_normal([in_size, out_size]), name='w1')
            w2 = tf.Variable(tf.random_normal([in_size, out_size]), name='w2')
            w3 = tf.Variable(tf.random_normal([in_size, out_size]), name='w3')
            b = tf.Variable(tf.zeros([1,out_size]) + 0.1, name='b')
            Wx_plus_b = tf.matmul(input1, w1) + tf.matmul(input2, w2) + tf.matmul(input3, w3) + b
            if activation_function is None:
                outputs = Wx_plus_b
            else:
                outputs = activation_function(Wx_plus_b)
            tf.summary.histogram( '/output', outputs)
            return outputs
train_x1, train_x2, train_x3, train_y = read_file("train.csv")
test_id, test_x1, test_x2, test_x3 = read_file("test.csv")


train_x1 = train_x1 / 100
test_x1 = test_x1 / 100
#train_x2 = train_x2 / 10
#test_x2 = test_x2 / 10
train_y = train_y / 10000000

batch_size = 128
epochs = 238

x1 = tf.placeholder(tf.float32,[None,1],name='x1')
x2 = tf.placeholder(tf.float32,[None,1],name='x2')
x3 = tf.placeholder(tf.float32,[None,1],name='x3')
ys = tf.placeholder(tf.float32,[None,1],name='ys')

l1 = add_layer(x1, x2, x3, 1, 30, n_layer=1, activation_function = tf.nn.relu)
l2 = add_layer(l1, None, None, 30, 30, n_layer=2, activation_function = tf.nn.tanh)
predition = add_layer(l2, None, None, 30, 1, n_layer=3, activation_function = tf.nn.softsign)

'''
l1 = add_layer(x1, x2, x3, 1, 30, n_layer=1, activation_function = tf.nn.relu)
#l1 = add_layer(x1, x2, x3, 1, 30, n_layer=1, activation_function = tf.nn.tanh)
predition = add_layer(l1, None, None, 30, 1, n_layer=2, activation_function = None)
'''

with tf.name_scope('loss'):
    loss = tf.reduce_mean(tf.reduce_sum(tf.square(ys - predition),reduction_indices=[1]))
    tf.summary.scalar('loss', loss)

with tf.name_scope('train'):
    train_step = tf.train.GradientDescentOptimizer(0.1).minimize(loss)

with tf.Session() as sess:
    init = tf.global_variables_initializer()
    sess.run(init)
    merged = tf.summary.merge_all()
    writer = tf.summary.FileWriter("logs/", sess.graph)

    for i in range(epochs):
        r = i * batch_size
        l = r + 128
        done = False
        if(l > train_x1.shape[0]):
            l = train_x1.shape[0]
        sess.run(train_step,feed_dict={x1:train_x1[r:l], x2:train_x2[r:l], x3:train_x3[r:l],ys:train_y[r:l]})
        if i % 20 == 0:
            res = sess.run(merged,feed_dict={x1:train_x1[r:l], x2:train_x2[r:l], x3:train_x3[r:l], ys:train_y[r:l]})
            writer.add_summary(res, i)
        if done:
            res = sess.run(merged,feed_dict={x1:train_x1[r:l], x2:train_x2[r:l], x3:train_x3[r:l], ys:train_y[r:l]})
            writer.add_summary(res, i)
            break

    res = sess.run(predition,feed_dict={x1:test_x1, x2:test_x2, x3:test_x3})
    res = res * 10000000
    res = np.absolute(res)
    write_file(test_id,res)
