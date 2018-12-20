from __future__ import division, print_function, absolute_import

import tensorflow as tf
import keras.backend as K

def refine_boxes(deltas, anchors):
    ''' Apply deltas to anchors
    Input:
        deltas: [1,-1, 4]
        anchors: [1,-1, 4]
    '''
    box_xy = deltas[:,:,0:2]*anchors[:,:,2:4] + anchors[:,:, 0:2]
    box_wh = K.exp(deltas[:,:,2:4]) *anchors[:,:,2:4]
    
    return K.concatenate([box_xy, box_wh],axis = -1)

def eval_graph(*args, config=None):
    '''
    Input:
        box_map: [batch, 19, 19, 5*4]. Float32. 
        class_map: [batch, 19, 19, 5*2]. Float32.
        anchors: [batch, 19,19, 5,4]. Int 16. Absolute value coordinate with input_shape
    Return:
        max_delat: [dx, dy, dw, dh]
        max_anchor: [x, y, w, h]
    '''
    box_map = args[0]
    class_map = args[1]
    anchors = args[2]
    
    # When evaluating batch size must be 1.!!!!!!
    # Change pytorch type data to tensorflow data
    # reshape to -1, so argmax can be used
    box_map = K.reshape(box_map, (-1,4,5))
    box_map = tf.transpose(box_map,(0,2,1))
    box_map = tf.reshape(box_map,(1,-1,4))
    
    class_map = K.reshape(class_map, (-1,2,5))
    class_map = tf.transpose(class_map,(0,2,1))
    class_map = tf.reshape(class_map,(1,-1,2))
    
    anchors = K.reshape(anchors, (1,-1,4))
    
    refined_box = refine_boxes(box_map, anchors)
    # Softmax activation
    class_map = K.softmax(class_map, -1)
    class_map = class_map[...,1]
    return [refined_box,class_map]
