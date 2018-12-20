from __future__ import division, print_function, absolute_import

import numpy as np

def parse_outputs(boxes,scores,scale_z,config):
    ''' Copied from original DaSiameseRPN
    '''
    target_xy = config.outputs['target_xy']
    target_wh = config.outputs['target_wh']*scale_z

    def change(r):
        return np.maximum(r, 1./r)

    def sz(w, h, context_amout = 0.5):
        pad = (w + h) * context_amout
        sz2 = (w + pad) * (h + pad)
        return np.sqrt(sz2)

    def sz_wh(wh):
        pad = (wh[0] + wh[1]) * 0.5
        sz2 = (wh[0] + pad) * (wh[1] + pad)
        return np.sqrt(sz2)
    
    # Scale size penalty
    s_c = change(sz(boxes[:,2], boxes[:,3]) / (sz_wh(target_wh)))
    # W/h ratio penalty
    r_c = change((target_wh[0] / target_wh[1]) / (boxes[:,2] / boxes[:,3]))

    penalty = np.exp(-(r_c * s_c - 1.) * config.penalty_k)
    pscore = penalty * scores

    # Window float
    pscore = pscore * (1 - config.window_influence) + config.window * config.window_influence
    best_pscore_id = np.argmax(pscore)

    target = boxes[best_pscore_id] / scale_z
    target_wh = target_wh / scale_z
    lr = penalty[best_pscore_id] * scores[best_pscore_id] * config.lr

    res_x = target[0] + target_xy[0]
    res_y = target[1] + target_xy[1]

    res_w = target_wh[0] * (1 - lr) + target[2] * lr
    res_h = target_wh[1] * (1 - lr) + target[3] * lr

    target_xy = np.array([res_x, res_y])
    target_wh = np.array([res_w, res_h])

    return target_xy, target_wh, scores[best_pscore_id]
