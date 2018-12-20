from __future__ import division, print_function, absolute_import

import tensorflow as tf
import keras.layers as KL
import keras.models as KM

def alexnet_graph(imgs):
    x = KL.Conv2D(192,kernel_size=11,strides=2,name = 'conv0')(imgs)
    
    x = KL.BatchNormalization(name = 'bn0')(x)
    x = KL.Activation('relu',name='relu0')(x)
    x = KL.MaxPool2D(3, strides=2, name= 'pool0')(x)
    
    x = KL.Conv2D(512,kernel_size=5,name = 'conv1')(x)
    x = KL.BatchNormalization(name='bn1')(x)
    x = KL.Activation('relu',name='relu1')(x)
    x = KL.MaxPool2D(3, strides=2,name='pool1')(x)
    
    x = KL.Conv2D(768,kernel_size=3,name='conv2')(x)
    x = KL.BatchNormalization(name='bn2')(x)
    x = KL.Activation('relu',name='relu2')(x)

    x = KL.Conv2D(768,kernel_size=3,name='conv3')(x)
    x = KL.BatchNormalization(name='bn3')(x)
    x = KL.Activation('relu',name='relu3')(x)
    
    x = KL.Conv2D(512,kernel_size=3,name='conv4')(x)
    x = KL.BatchNormalization(name='bn4')(x)
    
    return [x]
def build_encoder(input_shape = (None,None,3)):
    
    input_imgs = KL.Input(shape=input_shape,
                                 name="input_imgs")
    outputs = alexnet_graph(input_imgs)
    return KM.Model([input_imgs], outputs, name="Alexnet_encoder")

if __name__ == '__main__':
    model = build_encoder()
    model.summary()
