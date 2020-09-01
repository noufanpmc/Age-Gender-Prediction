# -*- coding: utf-8 -*-
"""
Created on Tue Feb 19 00:50:33 2019

@author: SKT
"""

from scipy import misc
import numpy as np
from keras import backend as K

from SSRNET_model import SSR_net
from config import AGE_WEIGHTS

IMG_SIZE = 64
    
def load_model(weights = AGE_WEIGHTS):    
    stage_num = [3,3,3]
    lambda_local = 1
    lambda_d = 1
    model = SSR_net(IMG_SIZE,stage_num, lambda_local, lambda_d)()
    model.load_weights(weights)
    return model

def preprocess(img):
    crop = misc.imresize(img, (IMG_SIZE, IMG_SIZE), interp='bilinear')
    
    #rgb to bgr
    image = crop[...,::-1]

    return image   

def predict(image, path=True):
    
    #load image if path given
    if path:
        image = misc.imread(image)
     
    image = preprocess(image)
    faces = np.empty((1, IMG_SIZE, IMG_SIZE, 3))
    faces[0,:,:,:] = image

    #reset graph
    K.clear_session()
    
    model = load_model()
    
    pred = model.predict(faces)
    
    return int(pred[0][0])
