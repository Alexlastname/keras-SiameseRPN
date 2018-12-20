from __future__ import division, print_function, absolute_import

import tensorflow as tf
import keras.layers as KL

class CONV(KL.Layer):
    def __init__(self,config,padding = 'VALID',**kwargs):
        self.config = config
        self.padding = padding
        super(__class__,self).__init__(**kwargs)
    
    def call(self, inputs, **kwargs):
        img = inputs[0]
        templates = inputs[1]
        
        self.temp_shps = tf.shape(templates)
        self.img_shps = tf.shape(img)
        # Reshape templates to kernels
        templates = tf.squeeze(templates,axis=0)
        templates = tf.reshape(templates,(self.temp_shps[1],self.temp_shps[2],-1,self.config.encoder_out_filter))
        templates = tf.transpose(templates, (0,1,3,2))
        #out_dim = tf.shape(templates)[0]
        
        out = tf.nn.conv2d(img, templates, strides=(1,1,1,1), padding = self.padding,)
        
        return out
    def compute_output_shape(self, input_shape):
        if self.padding =='VALID':
            return (None,input_shape[0][1]-input_shape[1][1]+1,input_shape[0][2]-input_shape[1][2]+1,
                    int(input_shape[1][-1]/self.config.encoder_out_filter))
        else:
            return (None,input_shape[0][1],input_shape[0][2],
                    int(input_shape[1][-1]/self.config.encoder_out_filter))


class siamese_conv(KL.Layer):
    '''
    template : [batch, 4, 4, anchors*256*(2,4)]
    img : [batch, 20, 20, 256]
    '''
    def __init__(self,config,padding = 'VALID',**kwargs):
        self.config = config
        self.padding = padding
        super(__class__,self).__init__(**kwargs)
    
    def call(self, inputs, **kwargs):
        """Tensorflow can't directly convolve different kernels, use depthwise_conv2d
            to get around.
        """
        img = inputs[0]
        templates = inputs[1]
        
        self.temp_shps = tf.shape(templates)
        self.img_shps = tf.shape(img)
        
        templates = tf.reshape(templates,(self.temp_shps[0],self.temp_shps[1],
                                          self.temp_shps[2],self.config.encoder_out_filter,-1))
        out_dim = tf.shape(templates)[-1]
        
        F = tf.transpose(templates,(1,2,0,3,4))
        F = tf.reshape(F,(self.temp_shps[1],self.temp_shps[2],self.temp_shps[0]*self.config.encoder_out_filter,out_dim))
        
        img = tf.transpose(img,(1,2,0,3))
        img = tf.reshape(img,(1,self.img_shps [1],self.img_shps [2],-1))
        
        out = tf.nn.depthwise_conv2d(img,F,(1,1,1,1),padding=self.padding)
        
        if self.padding == 'VALID':
            out = tf.reshape(out,(self.img_shps [1]-self.temp_shps[1]+1,self.img_shps [2]-self.temp_shps[2]+1,
                                  self.img_shps [0],self.config.encoder_out_filter,out_dim))
        else:
            out = tf.reshape(out,(self.img_shps [1],self.img_shps [2],self.img_shps [0],self.config.encoder_out_filter,out_dim))
        
        out = tf.transpose(out,(2,0,1,3,4))
        out = tf.reduce_sum(out,axis = 3)
        
        return out
    def compute_output_shape(self, input_shape):
        #print('Computing siamese output shape',input_shape)
        if self.padding =='VALID':
            return (None,input_shape[0][1]-input_shape[1][1]+1,input_shape[0][2]-input_shape[1][2]+1,
                    int(input_shape[1][-1]/self.config.encoder_out_filter))
        else:
            return (None,input_shape[0][1],input_shape[0][2],
                    int(input_shape[1][-1]/self.config.encoder_out_filter))
