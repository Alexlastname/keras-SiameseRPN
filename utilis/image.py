from __future__ import division, print_function, absolute_import

import cv2
import numpy as np

def resize_box(img, out_shape = (512,512), pad_val = np.zeros((3,))):
    ''' 
    '''
    ih,iw = img.shape[:2]
    h,w = out_shape
    scale = min((h/ih),(w/iw))
            
    nh = int(ih*scale)
    nw = int(iw*scale)
    img = cv2.resize(img,(nw,nh),cv2.INTER_AREA)
    
    out_img = np.zeros(out_shape+(3,))
    
    dh = (h-nh)//2
    dw = (w - nw)//2
    
    out_img[:,:] = np.around(pad_val)
    out_img[dh:dh+nh,dw:dw+nw,:] = img
    
    return out_img

def get_subwindow_tracking(im, pos, model_sz, original_sz, avg_chans):
    ''' Copied from DasiameseRPN 
    Input:
        im: [None, None, (B,G,R)]. Int. Oringal Image with uncertain resolution
        pos: [x, y]. Int. Last spotted tracking item's center
        model_sz: []. Int. Size of output picture in img resolution
        original_sz: []. Int. Shape of network inputs.
        avg_chans: [(B,G,R)]. Float. Avg values of template image each channel.
    Return:
        im_patch: [model_sz, model_sz, 3]. Int. Resized input image. 
    '''
    #print(pos, model_sz, original_sz, avg_chans)
    if isinstance(pos, float):
        pos = [pos, pos]
    sz = original_sz
    im_sz = im.shape
    c = (original_sz+1) // 2
    context_xmin = round(pos[0] - c)  # floor(pos(2) - sz(2) / 2);
    context_xmax = context_xmin + sz - 1
    context_ymin = round(pos[1] - c)  # floor(pos(1) - sz(1) / 2);
    context_ymax = context_ymin + sz - 1
    left_pad = int(max(0., -context_xmin))
    top_pad = int(max(0., -context_ymin))
    right_pad = int(max(0., context_xmax - im_sz[1] + 1))
    bottom_pad = int(max(0., context_ymax - im_sz[0] + 1))

    context_xmin = context_xmin + left_pad
    context_xmax = context_xmax + left_pad
    context_ymin = context_ymin + top_pad
    context_ymax = context_ymax + top_pad
    #print(context_xmin,context_xmax,context_ymin,context_ymax)
    # zzp: a more easy speed version
    r, c, k = im.shape
    if any([top_pad, bottom_pad, left_pad, right_pad]):
        te_im = np.zeros((r + top_pad + bottom_pad, c + left_pad + right_pad, k), np.uint8)  # 0 is better than 1 initialization
        te_im[top_pad:top_pad + r, left_pad:left_pad + c, :] = im
        if top_pad:
            te_im[0:top_pad, left_pad:left_pad + c, :] = avg_chans
        if bottom_pad:
            te_im[r + top_pad:, left_pad:left_pad + c, :] = avg_chans
        if left_pad:
            te_im[:, 0:left_pad, :] = avg_chans
        if right_pad:
            te_im[:, c + left_pad:, :] = avg_chans

        im_patch_original = te_im[int(context_ymin):int(context_ymax + 1), int(context_xmin):int(context_xmax + 1), :]
    else:
        im_patch_original = im[int(context_ymin):int(context_ymax + 1), int(context_xmin):int(context_xmax + 1), :]

    if not np.array_equal(model_sz, original_sz):
        im_patch = cv2.resize(im_patch_original, (model_sz, model_sz))
    else:
        im_patch = im_patch_original
    return im_patch
