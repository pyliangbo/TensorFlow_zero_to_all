# -*- coding: utf-8 -*-
"""
Created on Tue Apr  2 18:45:57 2019

@author: 梁博
"""

import numpy as np
import tensorflow as tf
import sklearn.preprocessing as prep
from tensorflow.examples.tutorials.mnist import input_data

if __name__=='__main__':
#    mnist=input_data.read_data_sets('Mnist_data/',one_hot=True)
    sess=tf.InteractiveSession()
    in_units=784
    h1_units=300
    W1=tf.Variable(tf.truncated_normal([in_units,h1_units],stddev=0.1))
    b1=tf.Variable(tf.zeros([h1_units]))
    W2=tf.Variable(tf.zeros([h1_units,10]))
    b2=tf.Variable(tf.zeros([10]))
    
    x=tf.placeholder(tf.float32,[None,in_units])
#keep_prob 保留数据而不置为0的比例
    keep_prob=tf.placeholder(tf.float32)
    hidden1=tf.nn.relu(tf.matmul(x,W1)+b1)
    hidden1_drop=tf.nn.dropout(hidden1,keep_prob)
    y=tf.nn.softmax(tf.matmul(hidden1_drop,W2)+b2)
    
    y_=tf.placeholder(tf.float32,[None,10])
    cross_entropy=tf.reduce_mean(-tf.reduce_sum(y_*tf.log(y),reduction_indices=[1]))
    train_step=tf.train.AdagradOptimizer(0.3).minimize(cross_entropy)
    
    tf.global_variables_initializer().run()
    for i in range(5000):
        batch_xs,batch_ys=mnist.train.next_batch(100)
        train_step.run({x:batch_xs,y_:batch_ys,keep_prob:0.75})

    correct_predication=tf.equal(tf.argmax(y,1),tf.argmax(y_,1))
    accuracy=tf.reduce_mean(tf.cast(correct_predication,tf.float32))
    print(accuracy.eval({x:mnist.test.images,y_:mnist.test.labels,keep_prob:1.0}))       
        
        
        
        
        
        