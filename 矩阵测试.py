import numpy as np

import tensorflow as tf
import csv
from sklearn.preprocessing import LabelEncoder
from keras.utils import np_utils

# 样本个数
sample_num = 2
# 设置迭代次数
epoch_num =300
# 设置一个批次中包含样本个数
batch_size = 1
# 计算每一轮epoch中含有的batch个数
batch_total = int(sample_num / batch_size) + 1


encoder = LabelEncoder()
sess = tf.Session()

a=[[1,2,3],[2,3,4]]
a=np.array(a)

b=[1,0]


lr = tf.Variable(0.001, dtype=tf.float32)

a = np.reshape(a,[2,1,3,1])
b = encoder.fit_transform(b)
b = np_utils.to_categorical(b)
x_vals=a

dummy_y=b

xx=a
dummy_ytest=b


def get_batch_data(x_v,y_v,batch_size=batch_size):
    images = x_v
    label = y_v
    # 数据类型转换为tf.float32
    images = tf.cast(images, tf.float32)
    label = tf.cast(label, tf.int32)

    # 从tensor列表中按顺序或随机抽取一个tensor
    input_queue = tf.train.slice_input_producer([images, label], shuffle=False)

    image_batch, label_batch = tf.train.batch(input_queue, batch_size=batch_size, num_threads=1, capacity=64)

    return image_batch, label_batch



def compute_accuracy(v_xs, v_ys):
    global prediction
    global acc
    y_pre = sess.run(prediction, feed_dict={xs: v_xs, keep_prob: 1.0})
    yp = np.argmax(y_pre, 1)
    yt = np.argmax(v_ys, 1)
    confuse_martix = sess.run(tf.convert_to_tensor(tf.confusion_matrix(yt, yp)))
    correct_prediction = tf.equal(tf.argmax(y_pre,1), tf.argmax(v_ys,1))

    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

    result = sess.run(accuracy, feed_dict={xs: v_xs, ys: v_ys,keep_prob:1.0})
    return result,confuse_martix


def conv2d(x,w):
    return tf.nn.conv2d(x,w,strides=[1,1,1,1],padding="SAME")
def pool(x):
    return tf.nn.max_pool(x,ksize=[1,1,1,1],strides=[1,1,1,1],padding="SAME")
def weight(shape):
    intial = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(intial)
def baise(shape):
    intitial = tf.constant(0.1, shape=shape)
    return tf.Variable(intitial)

# input = np.reshape(x_vals,[90,1,41,1])

xs = tf.placeholder(tf.float32,[None,None,None,None])
ys = tf.placeholder(tf.float32,[None,None])
keep_prob = tf.placeholder(tf.float32)



#卷积
w1 = weight([1,5,1,8])
b1 = baise([8])
conv1 = conv2d(xs,w1)
h_conv1 = tf.nn.tanh(conv1 + b1)
pool1 = pool(h_conv1)


w2 = weight([1,5,8,16])
b2 = baise([16])
conv2 = conv2d(pool1,w2)
h_conv2 = tf.nn.tanh(conv2 + b2)
pool2 = pool(h_conv2)

# w3 = weight([1,5,16,32])
# b3 = baise([32])
# conv3 = conv2d(pool2,w3)
# h_conv3 = tf.nn.tanh(conv3 + b3)
# pool3 = pool(h_conv3)
# # #
# w4 = weight([1,5,32,64])
# b4 = baise([64])
# conv4 = conv2d(pool3,w4)
# h_conv4 = tf.nn.tanh(conv4 + b4)
# pool4 = pool(h_conv4)
# #
# w5 = weight([1,16,32,80])
# b5 = baise([80])
# conv5 = conv2d(pool4,w5)
# h_conv5 = tf.nn.tanh(conv5 + b5)
# pool5 = pool(h_conv5)




#全连接
w8 = weight([3*16,1000])
b8 = baise([1000])
h_pool = tf.reshape(pool2,[-1,3*16])
h_fc1 = tf.nn.tanh(tf.matmul(h_pool,w8 ) + b8)
h_fc1_drop = tf.nn.dropout(h_fc1,keep_prob)

w9 = weight([1000,500])
b9 = baise([500])
# h_pool9 = tf.reshape(h_fc1,[-1,500])
h_fc19 = tf.nn.tanh(tf.matmul(h_fc1,w9 ) + b9)
h_fc1_drop9 = tf.nn.dropout(h_fc19,keep_prob)

#输出
W_fc2 = weight([500,2])
b_fc2 = baise([2])
prediction = tf.nn.softmax(tf.matmul(h_fc1_drop9,W_fc2) + b_fc2)


correct_prediction = tf.equal(tf.argmax(prediction,1), tf.argmax(ys,1))
with tf.name_scope('acc'):
   accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
   acctest = tf.summary.scalar('acctest',accuracy)
   acctrain = tf.summary.scalar('acctrain',accuracy)

with tf.name_scope('trainloss'):
 cross_entropy= -tf.reduce_sum(ys*tf.log(prediction))
 # cross_entropy = -class0_weight * tf.reduce_sum(ys[:, 0] * tf.log(prediction[:, 0])) - class1_weight * tf.reduce_sum(ys[:, 1] * tf.log(prediction[:, 1]))
 trainloss =  tf.summary.scalar('trainloss',cross_entropy)
 testloss = tf.summary.scalar('testloss', cross_entropy)

# gloabl_steps = tf.Variable(0,trainable=False)
# learing_rate = tf.train.exponential_decay(0.1,gloabl_steps,batch_total,0.99,staircase=True)


train_step=tf.train.AdamOptimizer(0.0001).minimize(cross_entropy)

# x_batch, y_batch = get_Batch(input, dummy_y, 13997)


writer = tf.summary.FileWriter("123/", sess.graph)
image_batch, label_batch = get_batch_data(x_vals,dummy_y,batch_size=batch_size)
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(sess, coord)
    try:
        for i in range(epoch_num):  # 每一轮迭代
            print('第',i,'次')

            # a = compute_accuracy(xx,dummy_ytest)
            # b = compute_accuracy(xx21,dummy_21ytest)

            # print('/n')
            # print(b)

            for j in range(batch_total):  # 每一个batch
                # 获取每一个batch中batch_size个样本和标签

               x_train ,y_train  = sess.run([image_batch, label_batch])

              # print('第',j,'次')
               sess.run(train_step, feed_dict={xs: x_train, ys: y_train, keep_prob: 0.5})
            a, c = compute_accuracy(xx, dummy_ytest)
            print(a, c)


        # tp = c[1][1]
        # fp = c[1][0]
        # fn = c[0][1]
        # tn = c[0][0]
        # precision = tp / (tp + fp)
        # recall = tp / (tp + fn)
        # f1 = 2 * (precision * recall / (precision + recall))
        # print(precision, recall, f1)




        # print(sess.run(cross_entropy, feed_dict={xs: x_train, ys: y_train, keep_prob: 1.0}))
        # print(sess.run(cross_entropy1, feed_dict={xs: xx, ys: dummy_ytest, keep_prob: 1.0}))
    except tf.errors.OutOfRangeError:
        print("done")
    finally:
        coord.request_stop()
    coord.join(threads)

# for i in range(20000):
#     # data, label = sess.run([x_batch, y_batch])
#     sess.run(train_step, feed_dict={xs:input, ys:dummy_y, keep_prob:0.6})
#
#     if i % 50 == 0:
#         print('准确率：',compute_accuracy(
#             xx, dummy_ytest))
#         print(sess.run(cross_entropy, feed_dict={xs: input, ys: dummy_y, keep_prob: 0.6}))
#         print(sess.run(cross_entropy1, feed_dict={xs: xx, ys: dummy_ytest, keep_prob: 0.6}))
#