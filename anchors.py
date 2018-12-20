from __future__ import division, print_function, absolute_import

import numpy as np
import tensorflow as tf
def generate_anchors(total_stride=8, scales=[8,], ratios=[0.33, 0.5, 1, 2, 3], score_size=19):
    '''Generating anchors
    params:
        total_stride: Int. Original img -> feature map will take total stride compress.
        scales: List(Int). Width of the bbox directly applied to feature map
        ratios: List(Float). Height:Width
        score_size: Int. Feature map size
    return:
        anchors: [score_size*score_size*anchor_num,(y,x,h,w)]
    '''
    anchor_num = len(ratios) * len(scales)
    anchor = np.zeros((anchor_num, 4),  dtype=np.float32)
    size = total_stride * total_stride
    count = 0
    for ratio in ratios:
        ws = int(np.sqrt(size / ratio))
        hs = int(ws * ratio)
        for scale in scales:
            wws = ws * scale
            hhs = hs * scale
            anchor[count, 0] = 0
            anchor[count, 1] = 0
            anchor[count, 2] = wws
            anchor[count, 3] = hhs
            count += 1
    #print('pre_tile:',anchor)
    anchor = np.expand_dims(anchor, 0)
    anchor = np.tile(anchor, (score_size * score_size,1,1))
    anchor = np.reshape(anchor,(score_size,score_size,anchor_num,4))
    #print(anchor.shape)
    # Zero-center
    ori = -(score_size/2)*total_stride
    yy = [ori + total_stride*dy for dy in range(score_size)]
    xx = [ori + total_stride*dx for dx in range(score_size)]
    
    yy,xx = np.meshgrid(xx,yy)
    grid = np.concatenate([np.expand_dims(yy,-1),np.expand_dims(xx,-1)],axis = -1)
    #print(grid)
    grid = np.expand_dims(grid, axis=2)
    anchor[:,:,:,:2] = grid
    #print(anchor[9,9,:])
    #anchor = np.reshape(anchor,(-1,4))
    #print(anchor.shape)
    return anchor

def main(_):
    anchors = generate_anchors()
    print(anchors)
    print(anchors.shape)
    
if __name__ == '__main__':main(0)
