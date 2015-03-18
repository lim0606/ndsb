# Kaggle NDSB compettision with deep learning

I finished the competition with 22nd position (with small difference(?) from the winning solution)! 
I participated this competition to wrap-up and study the recent deep learning techniques, so I want to share my experiences.

## Kaggle NDSB competition
- See following the competition page :) 
 http://www.kaggle.com/c/datasciencebowl
- the blog post of the winning solution
 http://benanne.github.io/2015/03/17/plankton.html
 
## My approach
I will explain my approach in following order (like the blog post of the winning solution!). I basically used Caffe (http://caffe.berkeleyvision.org/). 
1. Data preprocessing and Data augmentation
- Scaling
- Affine transformation (and resizing to 96 by 96)
- Offline data augmentation vs Online data augmentation
2. Network architectures
- Variations of cxx_net
- GoogLeNet for NDSB
3. Training
- Batch normalization
4. Inference
- Batch normalization
- Multiple inference from single input
- Model averaging (from a single network architecture)
- Model averaging 
5. Miscellany
- Change interpolation methods in image transformation (in Caffe) from linear interpolation to cubic. 
6. Final submission
- Model averaging of 18 GoogLeNets (for NDSB) and 14 cxx\_net variation models. 

### Data preprocessing and Data augmentation
- Scaling
- Affine transformation (and resizing to 96 by 96)
- Offline data augmentation vs Online data augmentation

  I simply re-scales pixel values from 0-255 to 0-1. Actually, I didn't compared the difference between results of the original images and re-scaled ones. However, re-scaling has been known for one of the standard pre-processing for image classification. I would be better to try zero-whitenning or some others, but I didin't. 
  
  From the beginning, I decided to use affine transformation for the data augmentation because of the observation from the winning solution of Kaggle CIFAR-10 competition (https://www.kaggle.com/c/cifar-10). I randomly stretch the image in x and y-axis with ratio 0.99~1.01, and rotate them. See `codes_for_caffe/image_data_affine_layer.cpp` and `models/train_val_googlenet_bn_var3.prototxt` for implementation.
  
  Since I want to apply affine transformation from the original images, I used `image_data_layer.cpp` as a base code. Eventhough using leveldb or lmdb data layer for input is much faster in genenel, it has not much difference when you use large network architectures (feedforward and backprop spends more time) and SSD for your disk. 
  
  A thread from the kaggle ndsb forum (http://www.kaggle.com/c/datasciencebowl/forums/t/12597/differences-between-real-time-augmentation-and-preprocess-augmentation/65673) discribed that the online data augumentation consistently made better results, so I decided to do that. I made a new layer for it :). See `codes_for_caffe/image_data_affine_layer.cpp` for the implementation. My personal comparisons between online and offline also showed that the online one produces consistently better results! 

### Network architectures
- Variations of cxx_net
- GoogLeNet for NDSB

I used following models; `models/train_model5_bn_realtime_aug.prototxt` and `models/train_val_googlenet_bn_var3.prototxt`

At first, I started from the model called cxx\_net that are shared by Bing Xi (http://www.kaggle.com/users/43581/bing-xu). See https://github.com/antinucleon/cxxnet/tree/master/example/kaggle_bowl

From the models, I increased the number of features in convolution layer, especially the first layer. I also applied the network in network layer (http://openreview.net/document/9b05a3bb-3a5e-49cb-91f7-0f482af65aea) to the variation of cxx_net. I used the caffe implementation of nin descibed in https://github.com/BVLC/caffe/wiki/Model-Zoo. I annotated the model as model5 in may files. See `models/train_model5_bn_realtime_aug.prototxt`

I also tried GoogLeNet for this competition because it is cool!!! I changed the size of convolution at the first convolution layer from 7 to 4 (since I thought the finer lower level features are required in this competition, and it was!), and changed some pooling and padding parameters. See `models/train_val_googlenet_bn_var3.prototxt`

There is a reason I used two different models in final submission. The predictions of GoogLeNet always produces better accuracy in validation data (randomly selected from 10% of training data) than 

### Training
- Batch normalization


### Inference
- Multiple inference from single input
- Model averaging (from a single network architecture)
- Model averaging 

### Miscellany
- Change interpolation methods in image transformation (in Caffe) from linear interpolation to cubic. 

### Final submission
- Model averaging of 18 GoogLeNets (for NDSB) and 14 cxx\_net variation models. 
I 
