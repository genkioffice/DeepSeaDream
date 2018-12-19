import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import tensorflow as tf
import sys
import os
import config


for path, di, file  in os.walk(config.PATH):
    CLASS = di
    break

# これらコードは、http://kivantium.hateblo.jp/entry/2015/11/18/233834から得られる
def inference(images_placeholder,keep_prob):
  def weight_variable(shape):
      initial = tf.truncated_normal(shape,stddev=0.1)
      return tf.Variable(initial)
  def bias_variable(shape):
      initial = tf.constant(0.1,shape=shape)
      return tf.Variable(initial)
  def conv2d(x,W):
      return tf.nn.conv2d(x,W,[1,1,1,1],padding='SAME')
  x_image = tf.reshape(images_placeholder,[-1,28,28,3])
  with tf.name_scope('conv1') as scope:
      W_conv1 = weight_variable([5,5,3,32])
      b_conv1 = bias_variable([32])
      h_conv1 = tf.nn.relu(conv2d(x_image,W_conv1) + b_conv1)

  with tf.name_scope('conv2') as scope:
      W_conv2 = weight_variable([5,5,32,64])
      b_conv2 = bias_variable([64])
      h_conv2 = tf.nn.relu(conv2d(h_conv1,W_conv2) + b_conv2)



  #全結合
  with tf.name_scope('zenketsu1') as scope:
      W_fc1 = weight_variable([7*7*64,1024])
      b_fc1 = bias_variable([1024])
      h_pool2_flat = tf.reshape(h_conv2,[-1,7*7*64])
      h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat,W_fc1) + b_fc1)
      #dropout
      h_fc1_drop = tf.nn.dropout(h_fc1,keep_prob)

  with tf.name_scope('zenketsu2') as scope:
      W_fc2 = weight_variable([1024,NUM_CLASSES])
      b_fc2 = bias_variable([NUM_CLASSES])

  with tf.name_scope('softmax') as scope:
      y_conv = tf.nn.softmax(tf.matmul(h_fc1_drop,W_fc2) + b_fc2)

  return y_conv
