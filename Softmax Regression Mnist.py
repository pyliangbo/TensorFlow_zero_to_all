# -*- coding: utf-8 -*-
"""
Created on Tue Apr  2 00:19:12 2019

@author: 梁博
"""
from tensorflow.contrib.learn.python.learn.datasets.mnist import read_data_sets
from tensorflow.examples.tutorials.mnist import input_data
import tensorflow as tf

mnist=input_data.read_data_sets('Mnist_data/',one_hot=True)
#
#print(mnist.train.images.shape,mnist.train.labels.shape)
#print(mnist.test.images.shape,mnist.test.labels.shape)
#print(mnist.validation.images.shape,mnist.validation.labels.shape)

sess=tf.InteractiveSession()
x=tf.placeholder(tf.float32,[None,784])
W=tf.Variable(tf.zeros([784,10]))
b=tf.Variable(tf.zeros([10]))
y=tf.nn.softmax(tf.matmul(x,W)+b)
#y=tf.nn.softmax(tf.add(tf.matmul(x,W),b))

y_=tf.placeholder(tf.float32,[None,10])
cross_entropy=tf.reduce_mean(-tf.reduce_sum(y_*tf.log(y),reduction_indices=[1]))
train_step=tf.train.GradientDescentOptimizer(0.5).minimize(cross_entropy)
tf.global_variables_initializer().run()
for i in range(100):
    batch_xs,batch_ys=mnist.train.next_batch(100)
    train_step.run({x:batch_xs,y_:batch_ys})
    
    
correct_predication=tf.equal(tf.argmax(y,1),tf.argmax(y_,1))
accuracy=tf.reduce_mean(tf.cast(correct_predication,tf.float32))
print(accuracy.eval({x:mnist.test.images,y_:mnist.test.labels}))