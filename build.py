from __future__ import division, print_function, absolute_import

from models.backbone import build_encoder
from models.RPN import CONV
from anchors import generate_anchors
#from models.head import loss_head
from models.eval_graph import eval_graph
from models.predict_func import parse_outputs
from utilis.data_format import data_crop


#import tensorflow as tf
import numpy as np
import keras.backend as K
import keras.layers as KL
import keras.models as KM

class Siamese_RPN():
    def __init__(self, mode, config, model_dir='./log'):
        """
        mode: Either "training" or "inference"
        config: A Sub-class of the Config class
        model_dir: Directory to save training logs and trained weights
        """
        assert mode in ['training', 'inference','inference_init']
        self.mode = mode
        self.config = config
        self.model_dir = model_dir
        if model_dir != None:
            self.set_log_dir()
        self.keras_model = self.build()
    def build(self):
        ##############
        # Inputs
        ##############
        if self.mode == 'inference':
            # When evaluating batch size must be 1.!!!!!!
            assert self.config.batch_size == 1
            
            inp_img = KL.Input(shape=self.config.instance_size,name='inp_img')
            # Generate anchors for every batch,
            anchors = generate_anchors(self.config.total_stride,
                                       self.config.scales,self.config.ratios,self.config.score_size)
            anchors = np.broadcast_to(anchors, (self.config.batch_size,)+anchors.shape)
            anchors = KL.Lambda(lambda x:K.variable(anchors),name = 'inp_anchors')(inp_img)
            #inp_template = KL.Input(batch_shape = (1,)+self.config.template_size, name='inp_template')
            cls_template = KL.Lambda(lambda x:K.variable(self.config.cls_template),name='cls_template')(inp_img)
            bbox_template = KL.Lambda(lambda x:K.variable(self.config.bbox_template),name = 'bbox_template')(inp_img)
        elif self.mode == 'inference_init':
            # Input template's batch size is nailed to 1. 
            inp_template = KL.Input(batch_shape = (1,)+self.config.template_size, name='inp_template')
        ###########################
        # Encoder
        ###########################
        self.encoder = build_encoder()
        if self.mode == 'inference_init':
            ###########
            # Init
            ###########
            cls_filters = 2*self.config.num_anchors*self.config.encoder_out_filter
            bbox_filters = 4*self.config.num_anchors*self.config.encoder_out_filter
            encoded_template = self.encoder(inp_template)
            cls_template = KL.Conv2D(cls_filters,(3,3),name='conv_cls1')(encoded_template)
            bbox_template = KL.Conv2D(bbox_filters,(3,3),name='conv_r1')(encoded_template)
            outputs = [cls_template,bbox_template]
            return KM.Model([inp_template],outputs,name = 'Siamese_init')
        
        elif self.mode == 'inference':
            ###################
            # Inference
            ###################
            encoded_img= self.encoder(inp_img)
            cls_img = KL.Conv2D(self.config.encoder_out_filter,(3,3),name='conv_cls2')(encoded_img)
            bbox_img = KL.Conv2D(self.config.encoder_out_filter,(3,3),name='conv_r2')(encoded_img)
            cls_out = CONV(self.config, name = 'cls_nn_conv')([cls_img,cls_template])
            bbox_out = CONV(self.config,name='box_nn_conv')([bbox_img,bbox_template])
            bbox_out = KL.Conv2D(4*self.config.num_anchors,1,name = 'regress_adjust')(bbox_out)
            
            outputs = KL.Lambda(lambda x:eval_graph(*x,config = self.config), name='Eval')([bbox_out, cls_out, anchors])
            return KM.Model([inp_img],outputs,name='Siamese_inference')
        
            
    def load_weights(self,filepath, by_name=True, skip_mismatch = False, 
                     reshape = False, exclude=None,verbose = False):
        import h5py
        import re
        from keras.engine import saving
        if h5py is None:
            raise ImportError('`load_weights` requires h5py.')
        keras_model = self.keras_model
        layers = keras_model.inner_model.layers if hasattr(keras_model, "inner_model")\
            else keras_model.layers
        if exclude != None:
            by_name = True
            layers = filter(lambda x: not re.match(exclude, x.name),layers)
        if verbose:
            print('[INFO] Loading following layers: ')
            for layer in layers:
                print('Layer:     ',layer.name)
        with h5py.File(filepath, mode='r') as f:
            if 'layer_names' not in f.attrs and 'model_weights' in f:
                f = f['model_weights']
            if by_name:
                saving.load_weights_from_hdf5_group_by_name(
                    f, layers, skip_mismatch=skip_mismatch,
                    reshape=reshape)
            else:
                saving.load_weights_from_hdf5_group(
                    f, layers, reshape=reshape)
    def inference_init(self,img):
        input_template = data_crop(img,self.config, mode = 'template')
        input_template = np.expand_dims(input_template, axis = 0)
        outputs = self.keras_model.predict(input_template)
        self.config.cls_template = outputs[0]
        self.config.bbox_template = outputs[1]
        print('[INFO] Store template feature_map')
    def predict(self, img):
        input_img,scale_size = data_crop(img, self.config, mode = 'instance')
        input_img = np.expand_dims(input_img, axis = 0)
        boxes, scores = self.keras_model.predict(input_img)
        
        boxes = np.squeeze(boxes, axis = 0)
        scores = np.squeeze(scores,axis = 0)

        xy,wh,score = parse_outputs(boxes, scores, scale_size, self.config)
        # Update outputs
        self.config.outputs['target_xy'] = xy
        self.config.outputs['target_wh'] = wh
        return xy,wh,score
def test():
    from config import Tracker_config
    
    config = Tracker_config()
    a = Siamese_RPN(mode='inference',config=config)
    a.load_weights('pretrained/baseline.h5',verbose=False)
    
if __name__ == '__main__':
    test()
