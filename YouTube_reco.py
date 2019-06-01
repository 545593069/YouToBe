#author: oger

from __future__ import print_function

import linecache
import time

import math
import numpy as np
import tensorflow as tf

train_file = "data/dbpedia_csv/dbpedia-train.train"
test_file = "data/dbpedia_csv/dbpedia-test.test"
label_dict = {}
sku_dict = {}

max_window_size = 1000
batch_size = 500
emb_size = 128

# Parameters
learning_rate = 0.01
training_epochs = 1
display_step = 1

# Network Parameters
# n_hidden_1 = 128 # 1st layer number of features
# # n_hidden_2 = 256 # 2nd layer number of features
n_hidden_1 = 1024 # 1st layer number of features
n_hidden_2 = 512 # 2nd layer number of features

print("Final Accuracy: ")
'''给数据加编号'''
def init_data(read_file):
    #0 is used for padding embedding
    label_cnt = 0
    sku_cnt = 1
    #FILE_OBJECT= open('order.log','rb')
    #f = open(read_file,'r')
    f= open(read_file,'r', encoding='UTF-8')
    for line in f:
        line = line.strip().split(' ')
        for i in line:
            if i.find('__label__') == 0:
                if i not in label_dict:
                    label_dict[i] = label_cnt
                    label_cnt += 1
            else:
                if i not in sku_dict:
                    sku_dict[i] = sku_cnt
                    sku_cnt += 1

'''将编号复制给Y，作为视频时长'''
def read_data(pos, batch_size, data_lst):
    batch = data_lst[pos:pos + batch_size]
    x = np.zeros((batch_size, max_window_size))
    mask = np.zeros((batch_size, max_window_size))
    y = []
    word_num = np.zeros((batch_size))
    line_no = 0
    for line in batch:
        line = line.strip().split(' ')
        y.append(label_dict[line[0]])
        col_no = 0
        for i in line[1:]:
            if i in sku_dict:
                x[line_no][col_no] = sku_dict[i]
                mask[line_no][col_no] = 1
                col_no += 1
            if col_no >= max_window_size:
                break
        word_num[line_no] = col_no
        line_no += 1

    return x, np.array(y).reshape(batch_size, 1), mask.reshape(batch_size, max_window_size, 1), word_num.reshape(batch_size, 1)

#========================
'''程序正式执行入口'''
init_data(train_file)
train_lst = linecache.getlines(train_file)  #按行读取数据集
n_classes = len(label_dict) #输入样本总数
print("Class Num: ", n_classes)

# Store layers weight & bias
'''创建存储层权重和偏差对象'''
weights = {
    'h1': tf.Variable(tf.random_normal([emb_size, n_hidden_1])),
    'h2': tf.Variable(tf.random_normal([n_hidden_1, n_hidden_2])),
    'out': tf.Variable(tf.random_normal([n_hidden_1, n_classes]))
}
biases = {
    'b1': tf.Variable(tf.random_normal([n_hidden_1])),
    'b2': tf.Variable(tf.random_normal([n_hidden_2])),
    'out': tf.Variable(tf.random_normal([n_classes]))
}

# Create model 以矩阵，权重和偏差对象构建分层模型
def multilayer_perceptron(x, weights, biases):
    # Hidden layer with RELU activation
    #x = tf.nn.dropout(x, 0.8)
    layer_1 = tf.add(tf.matmul(x, weights['h1']), biases['b1'])
    layer_1 = tf.nn.relu(layer_1)
    #dlayer_1 = tf.nn.dropout(layer_1, 0.5)
    #layer_2 = tf.add(tf.matmul(layer_1, weights['h2']), biases['b2'])
    #layer_2 = tf.nn.relu(layer_2)
    # Output layer with linear activation
    # out_layer = tf.matmul(layer_1, weights['out']) + biases['out']
    # return out_layer
    return layer_1

'''嵌入层embedding，将正负样本编码成向量'''
embedding = {
    'input':tf.Variable(tf.random_uniform([len(sku_dict)+1, emb_size], -1.0, 1.0))
    # 'output':tf.Variable(tf.random_uniform([len(label_dict)+1, emb_size], -1.0, 1.0))
}

emb_mask = tf.placeholder(tf.float32, shape=[None, max_window_size, 1])
word_num = tf.placeholder(tf.float32, shape=[None, 1])

x_batch = tf.placeholder(tf.int32, shape=[None, max_window_size])
y_batch = tf.placeholder(tf.int64, [None, 1])

input_embedding = tf.nn.embedding_lookup(embedding['input'], x_batch)
project_embedding = tf.div(tf.reduce_sum(tf.multiply(input_embedding,emb_mask), 1),word_num)    #得到正负样本的二维向量嵌入矩阵

# Construct model
pred = multilayer_perceptron(project_embedding, weights, biases)  #用嵌入矩阵，权重和偏差构造模型

'''使用nce损失函数优化精简正样本和负样本'''
# Construct the variables for the NCE loss
nce_weights = tf.Variable(
    tf.truncated_normal([n_classes, n_hidden_1],
                        stddev=1.0 / math.sqrt(n_hidden_1)))
nce_biases = tf.Variable(tf.zeros([n_classes]))

loss = tf.reduce_mean(tf.nn.nce_loss(weights=nce_weights,
                     biases=nce_biases,
                     labels=y_batch,
                     inputs=pred,
                     num_sampled=10,
                     num_classes=n_classes))

cost = tf.reduce_sum(loss) / batch_size
optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)
out_layer = tf.matmul(pred, tf.transpose(nce_weights)) + nce_biases

init = tf.global_variables_initializer()
#with tf.Session(config=tf.ConfigProto(log_device_placement=True)) as sess:
#开启TensorFlow数据流会话，
with tf.Session() as sess:
    sess.run(init)

    # Training cycle
    # start_time = time.time()
    total_batch = int(len(train_lst) / batch_size)
    print("total_batch of training data: ", total_batch)
    '''执行整个图，多次训练样本数据'''
    for epoch in range(training_epochs):
        avg_cost = 0.
        for i in range(total_batch):
            x, y, batch_mask, word_number = read_data(i * batch_size, batch_size, train_lst)
            _,c = sess.run([optimizer, cost], feed_dict={x_batch: x, emb_mask: batch_mask, word_num: word_number, y_batch: y})
            #print("Epoch %d Batch %d Elapsed time %fs" %(epoch, i, time.time() - start_time))
            # Compute average loss
            avg_cost += c / total_batch
            # correct_prediction = tf.equal(tf.argmax(out_layer, 1), tf.reshape(y_batch, [batch_size]))
            # accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))
            # print("Accuracy:", accuracy.eval({x_batch: x, y_batch: y, emb_mask: batch_mask, word_num: word_number}))

        # Display logs per epoch step
        if epoch % display_step == 0:
            print("Epoch:", '%04d' % (epoch + 1), "cost=", \
              "{:.9f}".format(avg_cost))

    # Test model
    correct_prediction = tf.equal(tf.argmax(out_layer, 1), tf.reshape(y_batch, [batch_size]))
    # Calculate accuracy
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))

    '''获取样本总个数，用于计算平均精度'''
    test_lst = linecache.getlines(test_file)
    total_batch = int(len(test_lst) / batch_size)
    final_accuracy = 0
    '''遍历计算生成的列表，计算平均精度'''
    for i in range(total_batch):
        x, y, batch_mask, word_number = read_data(i*batch_size, batch_size, test_lst)
        batch_accuracy = accuracy.eval({x_batch: x, y_batch: y, emb_mask: batch_mask, word_num: word_number})
        print("Batch Accuracy: ", batch_accuracy)
        final_accuracy += batch_accuracy
    print("Final Accuracy: ", final_accuracy * 1.0 / total_batch)
