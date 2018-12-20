from __future__ import division, print_function, absolute_import
import cv2
import numpy as np

from config import Tracker_config
from utilis.image import get_subwindow_tracking

def data_crop(img, config, mode = 'template'):
    ''' Resize img to input_shape, focus is on the last target_xy
    Input:
        img: [h, w, (B,G,R)]. Int16. 
    Return:
        inp_imgs: config.input_shape + [3,]. Int16.
        scale_size: Float. 
    '''
    assert mode in ['instance', 'template']
    
    crop_w = config.outputs['target_wh'][0] + config.context_amount*sum(config.outputs['target_wh'])
    crop_h = config.outputs['target_wh'][1] + config.context_amount*sum(config.outputs['target_wh'])
    
    if mode == 'template':
        original_sz = round(np.sqrt(crop_w*crop_h))
        config.avg_chans = np.mean(img,axis = (0,1))
        
        image = get_subwindow_tracking(img, 
                                       pos = config.outputs['target_xy'], 
                                       model_sz=config.template_size[0], 
                                       original_sz=original_sz, 
                                       avg_chans=config.avg_chans)
        return image
    elif mode =='instance':
        original_sz = np.sqrt(crop_w*crop_h)
        scale_size = config.template_size[0] / original_sz
        d_search = (config.instance_size[0] - config.template_size[0]) /2
        pad = d_search / scale_size
        original_sz  += 2*pad
        original_sz = np.around(original_sz)
        image = get_subwindow_tracking(img, 
                                       pos = config.outputs['target_xy'], 
                                       model_sz=config.instance_size[0], 
                                       original_sz=original_sz, 
                                       avg_chans=config.avg_chans)
        return image,scale_size

if __name__=='__main__':
    config = Tracker_config()
    config.outputs['target_xy'] = [2772,1299]
    config.outputs['target_wh'] = [312,828]
    
    img = cv2.imread('test.jpg')
    
    out = data_crop(img,config)
    cv2.imshow('F',out)
    cv2.imwrite('template.png',out)
    cv2.waitKey(1000)
    
    
    out = data_crop(img,config,mode = 'instance')
    cv2.imshow('F',out)
    cv2.imwrite('input.png',out)
    cv2.waitKey(0)
    
