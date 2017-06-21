from __future__ import print_function
import tensorflow as tf
import numpy as np
import csv
def read_file(file_name):
  with open(file_name, 'r') as f:
    if file_name == "train.csv":
      reader = csv.DictReader(f)
      col2  = []
      col3 = []
      col4 = []
      for row in reader:
        col2.append(float(row["railroad_station_walk_km"]))
        col3.append(float(row["public_transport_station_km"]))
        col4.append(float(row["price_doc"]))
      rows = len(col2)
      col2 = np.array(col2)
      col2 = col2.reshape(len(col2), 1)
      col3 = np.array(col3)
      col3 = col3.reshape(len(col3), 1)
      col4 = np.array(col4)
      col4 = col4.reshape(len(col4), 1)
      return col2, col3, col4, rows
    elif file_name == "test.csv":
      reader = csv.DictReader(f)
      col2 = []
      col3 = []
      for row in reader:
        col2.append(float(row["railroad_station_walk_km"]))
        col3.append(float(row["public_transport_station_km"]))
      col2 = np.array(col2)
      col2 = col2.reshape(len(col2), 1)
      col3 = np.array(col3)
      col3 = col3.reshape(len(col3), 1)
      rows = len(col2)
      return col2, col3, rows

train_x1, train_x2, train_y, train_rows = read_file("train.csv")
#test_x1, test_x2, test_rows = read_file("test_cut.csv")

def add_layer(input1, input2, in_size, out_size, n_layer,activation_function=None):
  layer_name = 'layer%s' % n_layer
  if input2 is None:
    with tf.name_scope(layer_name):
      with tf.name_scope('weights'):
        Weight = tf.Variable(tf.random_normal([in_size, out_size]), name='w')
        tf.summary.histogram(layer_name + '/weights', Weight)
      with tf.name_scope('biases'):
        biases = tf.Variable(tf.zeros([1,out_size]) + 0.1, name='b')
        tf.summary.histogram(layer_name + '/biases', biases)
      with tf.name_scope('Wx_plus_b'):
        Wx_plus_b = tf.matmul(input1, Weight) + biases
      if activation_function is None:
        outputs = Wx_plus_b
      else:
        outputs = activation_function(Wx_plus_b)
      return outputs
  else:
    with tf.name_scope(layer_name):
      with tf.name_scope('weights'):
        Weight_1 = tf.Variable(tf.random_normal([in_size, out_size]), name='w1')
        Weight_2 = tf.Variable(tf.random_normal([in_size, out_size]), name='w2')
        tf.summary.histogram(layer_name + '/weights', Weight_1)
        tf.summary.histogram(layer_name + '/weights', Weight_2)
      with tf.name_scope('biases'):
          biases = tf.Variable(tf.zeros([1,out_size]) + 0.1, name='b')
          tf.summary.histogram(layer_name + '/biases', biases)
      with tf.name_scope('Wx_plus_b'):
        Wx_plus_b = tf.matmul(input1, Weight_1) + tf.matmul(input2, Weight_2) + biases
      if activation_function is None:
        outputs = Wx_plus_b
      else:
        outputs = activation_function(Wx_plus_b)
      return outputs

train_y = train_y / 10000000

x1 =  tf.placeholder(tf.float32,[None,1])
x2 =  tf.placeholder(tf.float32,[None,1])
ys = tf.placeholder(tf.float32,[None,1])

l1 = add_layer(x1, x2, 1, 10, n_layer=1, activation_function = tf.nn.relu)
predition = add_layer(l1, None, 10, 1, n_layer=2, activation_function = None)

with tf.name_scope('loss'):
  loss = tf.reduce_mean(
    tf.reduce_sum(
        tf.square(ys - predition),reduction_indices=[1]
        )
    )
  tf.summary.scalar('loss', loss)

with tf.name_scope('train'):
  train_step = tf.train.GradientDescentOptimizer(0.1).minimize(loss)

init = tf.global_variables_initializer()
merged = tf.summary.merge_all()

sess = tf.Session()
sess.run(init)
writer = tf.summary.FileWriter("logs/", sess.graph)

for i in range(500):
  sess.run(train_step,feed_dict={x1:train_x1, x2:train_x2, ys:train_y})
  if i % 20 == 0:
    result = sess.run(merged,feed_dict={x1:train_x1, x2:train_x2, ys:train_y})
    print (result)
    writer.add_summary(result, i)
