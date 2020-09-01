# -*- coding: utf-8 -*-
"""
Created on Mon Feb 18 13:44:53 2019

@author: SKT
"""

import tensorflow as tf
from tensorflow.contrib.slim.python.slim.nets.inception_v3 import inception_v3_base
from tensorflow.contrib.layers import *
from scipy import misc
import numpy as np
import os

from config import GENDER_CHECKPOINT
RESIZE_FINAL = 227
LABEL_LIST =['male','female']

#model function
def inception_v3(nlabels, images, pkeep, is_training):

    batch_norm_params = {
        "is_training": is_training,
        "trainable": True,
        # Decay for the moving averages.
        "decay": 0.9997,
        # Epsilon to prevent 0s in variance.
        "epsilon": 0.001,
        # Collection containing the moving mean and moving variance.
        "variables_collections": {
            "beta": None,
            "gamma": None,
            "moving_mean": ["moving_vars"],
            "moving_variance": ["moving_vars"],
        }
    }
        
    weight_decay = 0.00004
    stddev=0.1
    weights_regularizer = tf.contrib.layers.l2_regularizer(weight_decay)
    with tf.variable_scope("InceptionV3", "InceptionV3", [images]) as scope:
        with tf.contrib.slim.arg_scope(
                [tf.contrib.slim.conv2d, tf.contrib.slim.fully_connected],
                weights_regularizer=weights_regularizer,
                trainable=True):
            with tf.contrib.slim.arg_scope(
                    [tf.contrib.slim.conv2d],
                    weights_initializer=tf.truncated_normal_initializer(stddev=stddev),
                    activation_fn=tf.nn.relu,
                    normalizer_fn=batch_norm,
                    normalizer_params=batch_norm_params):
                net, end_points = inception_v3_base(images, scope=scope)
                with tf.variable_scope("logits"):
                    shape = net.get_shape()
                    net = avg_pool2d(net, shape[1:3], padding="VALID", scope="pool")
                    net = tf.nn.dropout(net, pkeep, name='droplast')
                    net = flatten(net, scope="flatten")

    with tf.variable_scope('output') as scope:

        weights = tf.Variable(tf.truncated_normal([2048, nlabels], mean=0.0, stddev=0.01), name='weights')
        biases = tf.Variable(tf.constant(0.0, shape=[nlabels], dtype=tf.float32), name='biases')
        output = tf.add(tf.matmul(net, weights), biases, name=scope.name)

    return output

#get checkpoint from path
def get_checkpoint(checkpoint_path, requested_step=None, basename='checkpoint'):
    if requested_step is not None:

        model_checkpoint_path = '%s/%s-%s' % (checkpoint_path, basename, requested_step)
        if os.path.exists(model_checkpoint_path) is None:
            print('No checkpoint file found at [%s]' % checkpoint_path)
            exit(-1)
            print(model_checkpoint_path)
        print(model_checkpoint_path)
        return model_checkpoint_path, requested_step

    ckpt = tf.train.get_checkpoint_state(checkpoint_path)
    if ckpt and ckpt.model_checkpoint_path:
        # Restore checkpoint as described in top of this program
        print(ckpt.model_checkpoint_path)
        global_step = ckpt.model_checkpoint_path.split('/')[-1].split('-')[-1]

        return ckpt.model_checkpoint_path, global_step
    else:
        print('No checkpoint file found at [%s]' % checkpoint_path)
        exit(-1)
        
#normalize
def prewhiten(x):
    mean = np.mean(x)
    std = np.std(x)
    std_adj = np.maximum(std, 1.0/np.sqrt(x.size))
    y = np.multiply(np.subtract(x, mean), 1/std_adj)
    return y  

#preprocess                  
def preprocess(image): 
    image = prewhiten(image)    
    return image

def predict(image, path=True):
    
    #load image if path given
    if path:
        image = misc.imread(image)

    #reset tf
    tf.reset_default_graph()
          
    config = tf.ConfigProto(allow_soft_placement=True)
    with tf.Session(config=config) as sess:
        #labels list   
        label_list = LABEL_LIST
        nlabels = len(label_list)
        
        #checkpoint path
        checkpoint_path = GENDER_CHECKPOINT
      
        #getting model function
        model_fn = inception_v3
        
        #model creation
        images = tf.placeholder(tf.float32, [None, RESIZE_FINAL, RESIZE_FINAL, 3])
        logits = model_fn(nlabels, images, 1, False)
        softmax_output = tf.nn.softmax(logits)
                
        #initialize tf variables
        init = tf.global_variables_initializer()
        
        #restore checkpoint (load pre-trained model)
        model_checkpoint_path, global_step = get_checkpoint(checkpoint_path)
        saver = tf.train.Saver()
        saver.restore(sess, model_checkpoint_path)

        #preprocess image
        image = preprocess(image)

        #prediction
        batch_results = sess.run(softmax_output, feed_dict={images:[image]})
        output = batch_results[0]
        best = np.argmax(output)
        pred = label_list[best]
        prob = output[best]
            
        
    return pred, prob

