# keras-SiameseRPN
A keras re-implementation of DaSiameseRPN, weights converted from original pytorch version.

Temporarily, it can only tracking VOT dataset with pretrained weights. 

Train code and evaluate code will be updated soon.

# Requirement
  1. Keras >= 2.2.4
  2. Tensorflow-gpu >= 1.12.0
  3. opencv-python 

# How to start

  1. Download baseline.h5 from release or https://pan.baidu.com/s/1TIR8aqwBsV7N6CW89mreWg. Please it under dir '/pretrained'
  2. run "python vot_inference.py --dir [Path to your own vot dataset tracking sequence]"
  
