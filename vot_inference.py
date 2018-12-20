from __future__ import division, print_function, absolute_import

import cv2
import os
import numpy as np
import time
import tensorflow as tf

from build import Siamese_RPN
from config import Tracker_config

class Myconfig(Tracker_config):
    
    instance_size = (271,271,3)
    template_size = (127,127,3)
    
    batch_size = 1
    
config = Myconfig()

def init(img,xy,wh):
    config.outputs['target_xy'] = np.array(xy)
    config.outputs['target_wh'] = np.array(wh)
    with tf.device('/gpu:0'):
        network = Siamese_RPN(mode='inference_init', config=config, model_dir=None)
    network.load_weights('pretrained/baseline.h5', by_name = True )
    network.inference_init(img)
    with tf.device('/gpu:0'):
        network = Siamese_RPN(mode = 'inference',config = config, model_dir = None)
    network.load_weights('pretrained/baseline.h5',by_name = True)
    return network

def vot_inference(base_dir = '/home/xcy/append/dataset/VOT/vot2016/road'):
    gt_file = os.path.join(base_dir,'groundtruth.txt')
    f_g = open(gt_file)
    for i,line in enumerate(f_g.readlines()):
        image_file = os.path.join(base_dir,'{:08d}.jpg'.format(i+1))
        img = cv2.imread(image_file)
        x1,y1,x2,y2,x3,y3,x4,y4 = [float(x) for x in line.split(',')]
        
        x = int((x1+x2+x3+x4)/4)
        y = int((y1+y2+y3+y4)/4)
        w = abs(int(((x2-x1)+(x3-x4))/2))
        h = abs(int(((y3-y2)+(y4-y1))/2))
        if i == 0:
            net = init(img, xy=[x,y], wh=[w,h])
            print(x,y,w,h)
        x1  = x - w//2
        y1 = y - h//2
        x2 = x1 + w
        y2 = y1 + h
        cv2.rectangle(img,(x1,y1),(x2,y2),(255,255,255),2)
        if i > 0 :
            tic = time.time()
            xy,wh,scores = net.predict(img)
            print('Keras predict cost:',time.time() - tic)
            x1  = xy[0] - wh[0]//2
            y1 = xy[1] - wh[1]//2
            x2 = x1 + wh[0]
            y2 = y1 + wh[1]
            cv2.rectangle(img,(int(x1),int(y1)),(int(x2),int(y2)),(0,0,255),2)
            print('Outputs:',(int(x1),int(y1)),(int(x2),int(y2)),scores)
        
        cv2.imshow('frame',img)
        cv2.waitKey(1)
if __name__=='__main__':
    vot_inference()



