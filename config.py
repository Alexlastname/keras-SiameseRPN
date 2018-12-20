from __future__ import division, print_function, absolute_import
import numpy as np
class Tracker_config():
    
    encoder_out_filter = 512
    
    
    instance_size = (271,271,3)
    template_size = (127,127,3)
    
    context_amount = 0.5
    ###############
    # anchors
    ###############
    total_stride = 8
    
    context_amount = 0.5  # context amount for the exemplar
    ratios = [0.33, 0.5, 1, 2, 3]
    scales = [8, ]
    
    
    batch_size = 32
    TARGET_HIGH_THRESHOLD = 0.6
    TARGET_LOW_THRESHOLD = 0.3
    
    TRAIN_ROIS_PER_IMAGE = 64
    ROI_POSITIVE_RATIO = 0.25
    
    
    penalty_k = 0.055
    window_influence = 0.42
    lr = 0.295
    context_amount = 0.5
    
    ##################
    # state
    ##################
    outputs = {}
    outputs['target_xy'] = [2772,1299]
    outputs['target_wh'] = [312,828]
    
    avg_chans = np.zeros((3,))
    window_mode = 'cosine'

    
    def __init__(self):
        self.score_size = (self.instance_size[0]-self.template_size[0])//self.total_stride+1
        self.num_anchors = len(self.ratios) * len(self.scales)
        
        if self.window_mode == 'cosine':
            window = np.outer(np.hanning(self.score_size), np.hanning(self.score_size))
        elif self.window_mode == 'uniform':
            window = np.ones((self.score_size, self.score_size))
        window = np.expand_dims(window, axis = -1)
        window = np.tile(window,(1,1,self.num_anchors))
        self.window = window.flatten()
        print(window.shape)
        
        self.outputs['target_xy'] = np.array(self.outputs['target_xy'])
        self.outputs['target_wh'] = np.array(self.outputs['target_wh'])
        
        self.display()
    def display(self):
        """Display Configuration values."""
        print("\nConfigurations:")
        for a in dir(self):
            if not a.startswith("__") and not callable(getattr(self, a)):
                print("{:30} {}".format(a, getattr(self, a)))
        print("\n")
