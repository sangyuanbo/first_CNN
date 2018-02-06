import tensorflow as tf
import numpy as np

import matplotlib.pyplot as plt

def add_layer(inputs,in_size,out_size,activation_function=None):    #in_size是输入单位数量，out_size是输出数量
    Weights = tf.Variable(tf.random_normal([in_size,out_size]))

    biases = tf.Variable(tf.zeros([1,out_size]) + 0.1)

    Wx_plus_b = tf.matmul(inputs,Weights) + biases

    if activation_function is None:
        outputs = Wx_plus_b
    else:
        outputs = activation_function(Wx_plus_b)

    return outputs


xs = tf.placeholder(tf.float32,[None,1])
ys = tf.placeholder(tf.float32,[None,1])

#建立第一层输入层,1个输入10个输出
l1 = add_layer(xs,1,10,activation_function=tf.nn.relu)

#创建隐藏层,10个输入1个输出
prediction = add_layer(l1,10,1,activation_function=None)

# 计算loss
loss = tf.reduce_mean(tf.reduce_sum(tf.square(ys-prediction),reduction_indices=[1]))

# 进行训练train
train_step = tf.train.GradientDescentOptimizer(0.1).minimize(loss)

# 所有变量进行初始化
init = tf.initialize_all_variables()
sess = tf.Session()
sess.run(init)

# np.linspace等差数列
x_data = np.linspace(-1,1,300)[:,np.newaxis]
noise = np.random.normal(0,0.05,x_data.shape)
y_data = np.square(x_data) - 0.5 + noise

fig = plt.figure()
ax = fig.add_subplot(1,1,1)
ax.scatter(x_data,y_data)
plt.show()


for i in range(1000):
    sess.run(train_step,feed_dict={xs:x_data,ys:y_data})

    if i % 50 == 0:
        print(sess.run(loss,feed_dict={xs:x_data,ys:y_data}))


















