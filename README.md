# Kaggle NDSB compettision with deep learning

I finished the competition with 22nd position (with small difference(?) from the winning solution)! 
I participated this competition to wrap-up and study the recent deep learning techniques, so I want to share my experiences.

## Kaggle NDSB competition
- See following the competition page :) 
 http://www.kaggle.com/c/datasciencebowl
- the blog post of the winning solution
 http://benanne.github.io/2015/03/17/plankton.html
 
## My approach
I will explain my approach in following order (like the blog post of the winning solution!)
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

### Data preprocessing and Data augmentation
- Scaling
- Affine transformation (and resizing to 96 by 96)
- Offline data augmentation vs Online data augmentation
  I simply re-scales pixel values from 0-255 to 0-1. Actually, I didn't compared the difference between results of the original images and re-scaled ones. However, re-scaling has been known for one of the standard pre-processing for image classification. I would be better to try zero-whitenning or some others, but I didin't. 
  From the beginning, I decided to use affine transformation for the data augmentation because of the observation from the winning solution of Kaggle CIFAR-10 competition (https://www.kaggle.com/c/cifar-10). I randomly stretch the image in x and y-axis with ratio 0.99~1.01, and rotate them. see `codes_for_caffe/image_data_affine_layer.cpp` and 'models/train_val_googlenet_bn_var3.prototxt'. 

### Network architectures
- Variations of cxx_net

At first, I started from the model called cxx_net that are shared by Bing Xi (http://www.kaggle.com/users/43581/bing-xu). 
- see https://github.com/antinucleon/cxxnet/tree/master/example/kaggle_bowl

- GoogLeNet for NDSB

### Training
- Batch normalization


### Inference
- Multiple inference from single input
- Model averaging (from a single network architecture)
- Model averaging 

### Miscellany
- Change interpolation methods in image transformation (in Caffe) from linear interpolation to cubic. 

