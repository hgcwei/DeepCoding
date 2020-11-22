# import matplotlib
import glob
import os
import tensorflow as tf
import numpy as np
import time
import data_io

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

def rnn_basic(_x,_w1,_w2,_b1,_b2):
    logits = tf.matmul(tf.nn.sigmoid(tf.add(tf.matmul(_x, _w1), _b1)), _w2) + _b2
    _y = tf.nn.softmax(logits)
    return _y,logits

def rnn_output(_s1,_x,_s2,_w,_w2,_b1,_b2):
    _sxs = tf.concat([_s1,_x,_s2],1)
    output,lgts = rnn_basic(_sxs,_w,_w2,_b1,_b2)
    return output,lgts

def rnn_state(_e1,_e2, _w, _w2,_b1,_b2):
    _e12 = tf.concat([_e1,_e2],1)
    state,_ = rnn_basic(_e12,_w,_w2,_b1,_b2)
    return state

def networks_model2(s,_x1,_x2,_x3, _weight,_biase):

    w_pc = tf.concat([_weight['pre'],_weight['cur']],0)
    w_cn = tf.concat([_weight['cur'],_weight['next']],0)
    w_pcn = tf.concat([w_pc,_weight['next']],0)

    _s2 = rnn_state(s,_x1,w_pc,_weight['out'],_biase['b1'],_biase['out'])
    _s3 = rnn_state(_s2,_x2,w_pc,_weight['out'],_biase['b1'],_biase['out'])

    s2_ = rnn_state(_x3,s,w_cn,_weight['out'],_biase['b1'],_biase['out'])
    s1_ = rnn_state(_x2,s2_,w_cn,_weight['out'],_biase['b1'],_biase['out'])

    _y1,_ = rnn_output(s,_x1,s1_,w_pcn,_weight['out'],_biase['b1'],_biase['out'])
    _y2, lgts = rnn_output(_s2,_x2,s2_,w_pcn,_weight['out'],_biase['b1'],_biase['out'])
    _y3, _ = rnn_output(_s3,_x3,s,w_pcn,_weight['out'],_biase['b1'],_biase['out'])

    return _y1,_y2,_y3,lgts

def cnn_merge(_x):
    x1, x2 = tf.split(_x, [2, 1], axis=2)
    x21, x22 = tf.split(x2, [64, 90 - 64], axis=1)
    x21 = tf.reshape(x21, [-1, 64])

    conv31 = tf.layers.conv2d(inputs=x1, filters=30, kernel_size=[7, 2], padding="valid", activation=tf.nn.relu,
                              kernel_initializer=tf.truncated_normal_initializer(stddev=0.01))
    pool31 = tf.layers.max_pooling2d(inputs=conv31, pool_size=[2, 1], strides=2)
    drop31 = tf.layers.dropout(pool31, 0.5)

    conv32 = tf.layers.conv2d(inputs=drop31, filters=30, kernel_size=[3, 1], padding="valid",
                              activation=tf.nn.relu,
                              kernel_initializer=tf.truncated_normal_initializer(stddev=0.01))
    pool32 = tf.layers.max_pooling2d(inputs=conv32, pool_size=[2, 1], strides=2)
    drop32 = tf.layers.dropout(pool32, 0.5)

    re31 = tf.reshape(drop32, [-1, 20*30])

    co1 = tf.concat([re31,x21], axis=1)
    return co1


# --------------------------- 生成训练测试数据 -----------------------------------
path = 'D:/matlab_projs/DeepCoding2/mH/train/'
test_path = 'D:/matlab_projs/DeepCoding2/mH/test/'

# 将所有的图片resize成100*100
w = 9
h = 90
c = 1
pw = 1.0
n_epoch = 9
batch_size = 300


train_dir0 = path + '/0/'
train_dir1 = path + '/1/'

test_dir0 = test_path + '/0/'
test_dir1 = test_path + '/1/'

data, label = data_io.read_csv(path)

s = data.shape
print(s)
data = np.reshape(data,[s[0],h,w,1])
test_data, test_label = data_io.read_csv(test_path)
s0 = test_data.shape
test_data = np.reshape(test_data, [s0[0], h, w, 1])



# ------------------------------- 构建网络 -------------------------------------
# 占位符
x = tf.placeholder(tf.float32, shape=[None, h, w, c], name='x')
# x2 = tf.placeholder(tf.float32,shape=[None, h2],name = 'x2')
y_ = tf.placeholder(tf.int32, shape=[None, ], name='y_')
y0 = tf.placeholder(tf.float32, [None, 2], name= 'y0')

x1,x2,x3 =  tf.split(x,[3,3,3],axis=2)
co1= cnn_merge(x1)
co2= cnn_merge(x2)
co3= cnn_merge(x3)


weights = {

    'pre': tf.Variable(tf.truncated_normal([2, 20], stddev=0.01)),
    'cur': tf.Variable(tf.truncated_normal([664, 20], stddev=0.01)),
    'next': tf.Variable(tf.truncated_normal([2, 20], stddev=0.01)),
    'out': tf.Variable(tf.truncated_normal([20, 2], stddev=0.01))
}

biases = {
    'b1': tf.Variable(tf.zeros([20])),
    'out': tf.Variable(tf.zeros([2]))
}

y1,y2,y3,logits = networks_model2(y0,co1,co2,co3,weights,biases)

# y_hat, logits = cnn(x1)

loss = tf.reduce_mean(tf.nn.weighted_cross_entropy_with_logits(targets=tf.one_hot(y_,2),logits=logits,pos_weight=pw))

train_op = tf.train.AdamOptimizer(learning_rate=0.001).minimize(loss)
correct_prediction = tf.equal(tf.cast(tf.argmax(logits, 1), tf.int32), y_)
acc = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

# 训练和测试数据，可将n_epoch设置更大一些
sess = tf.InteractiveSession()
sess.run(tf.global_variables_initializer())
saver = tf.train.Saver()
start = time.clock()
for epoch in range(n_epoch):
    # training
    state0 = np.zeros((batch_size, 2))
    train_loss, train_acc, train_batch = 0, 0, 0
    for x_train_a, y_train_a in data_io.minibatches(data, label, batch_size, shuffle=True):
        _, err, ac = sess.run([train_op, loss, acc], feed_dict={x: x_train_a, y_: y_train_a,y0: state0})
        train_loss += err
        train_acc += ac
        train_batch += 1
    # validation
    val_loss, val_acc, val_batch = 0, 0, 0
    for x_val_a, y_val_a in data_io.minibatches(test_data, test_label, batch_size, shuffle=False):
        err, ac = sess.run([loss, acc], feed_dict={x: x_val_a, y_: y_val_a,y0: state0})
        val_loss += err
        val_acc += ac
        val_batch += 1
    print("(%d/%d) train loss: %f, train acc: %f, validation loss: %f ,validation acc: %f"
          % (n_epoch, epoch + 1, train_loss / train_batch, train_acc / train_batch, val_loss / val_batch,
             val_acc / val_batch))

end = time.clock()
print("time elaspe: %s" % (end - start))

s = np.zeros((s0[0], 2))
pred = np.zeros((s0[0], 2))
ss = np.zeros((1,2))
i = 0
for x_train_a, y_train_a in data_io.minibatches(test_data, test_label, 1, shuffle=False):
    pred[i, :] = sess.run(y2, feed_dict={x: x_train_a, y_: y_train_a,y0: ss})
    i = i + 1
saver.save(sess, 'DeepCoding/mH/mH_deepcoding.ckpt')
fpr, tpr, auc = data_io.eval_perf(test_label, pred[:, 1])
data_io.save_csv('DeepCoding/mH/mM_label.csv', test_label)
data_io.save_csv('DeepCoding/mH/mM_scores.csv', pred[:, 1])
data_io.save_csv('DeepCoding/mH/mM_test_results.csv', [tpr, 1 - fpr, auc, end-start])
print(tpr, 1 - fpr, auc)
sess.close()