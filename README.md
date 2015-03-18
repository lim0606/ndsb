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

1. First ordered list item
2. Another item
⋅⋅* Unordered sub-list. 
1. Actual numbers don't matter, just that it's a number
⋅⋅1. Ordered sub-list
4. And another item.


1. Data preprocessing and Data augmentation
* Scaling
* Affine transformation (and resizing to 96 by 96)
* Offline data augmentation vs Online data augmentation
2. Network architectures
⋅⋅* Variations of cxx_net
⋅⋅* GoogLeNet for NDSB
3. Training
⋅⋅* Batch normalization
4. Inference
⋅⋅* Batch normalization
⋅⋅* Multiple inference from single input
⋅⋅* Model averaging (from a single network architecture)
⋅⋅* Model averaging 
5. Miscellany
⋅⋅* Change interpolation methods in image transformation (in Caffe) from linear interpolation to cubic. 
6. Final submissions
⋅⋅* Model averaging of 18 GoogLeNets (for NDSB) and 14 cxx\_net variants. 

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

There is a reason I used two different models in final submission. The predictions of GoogLeNet always produced better accuracy in validation data (randomly selected from 10% of training data) than model5, but the GoogLeNet always had larger loss (multinomial log loss). Averaging the predictions of those two network architectures made much better accuracy and loss. The issue of loss and accuracy is also considered in the winning solution of this competition; see their blog post http://benanne.github.io/2015/03/17/plankton.html. 

### Training
- Batch normalization

The main focus of myself to participate this competition is to wrap-up new deep learning techniques, and I really wanted to try GoogLeNet with batch normalization (http://arxiv.org/abs/1502.03167). Fortunately, there was a caffe implemetation of the batch normalization for caffe master branch (https://github.com/ChenglongChen/batch_normalization). There is now caffe-dev branch version of it, but there wasn't when I started to use. So, I modified the codes. I further modifed his batch normalization layer because his version does not support batch normalization for inference. 

Anyway, batch normalization (with xavier initialization) was awesomely cool!!!!!! I personally tested the trainings of GoogLeNet with and without batch normaliztion for NDSB dataset. The convergence speeds of the batch nomalization applied one is way way faster as the paper reported. So, it seems true :) 

The xavier initialization was also important for maximizing the usefulness of batch normalization. For better optimizatino SGD, we should have stable and consistenty gradients, and the batch normalization make them possible by providing stable backpropagation signals via activity normalization. The xavier initialization is the result of the consideration to provide sufficient size of backprob signal by weight values. For further explanation, see this paper http://arxiv.org/abs/1502.01852.

### Inference
- Batch normalization
- Multiple inference from single input
- Model averaging (from a single network architecture)
- Model averaging 

As I explained above, I further modifed the batch normalization layer (https://github.com/ChenglongChen/batch_normalization) to use it in inference. See the paper (http://arxiv.org/abs/1502.03167) to understand the batch normalization in inference. 
See `codes_for_caffe/predict_bn.cpp` and `codes_for_caffe/test_bn.cpp` for the implementation of batch normalization in inference. 
See `models/train_val_googlenet_bn_var3.prototxt` for how I applied this in the model. 

During the competition, I saw this post http://www.kaggle.com/c/datasciencebowl/forums/t/12652/cnn, and followed the link in it https://github.com/msegala/Kaggle-National_Data_Science_Bowl. This implemetation described the multiple inference from single image with single model. This means that when you read an image, augment this data to multiple different image as you did in training, and averaging the predictions for each of them. I implemented this one for caffe. See `codes_for_caffe/avg_probs_layer.cpp`, `image_data_multi_infer_layer.cpp`, and `models/predict_googlenet_bn_var3_loss_avg.prototxt` fot the implementation. 

So, I applied 8 multiple inferences, and produced highly better results. Any further inference not make improvement.

Model averaging of different initialization for a single network architecture is well-known techniques, and I used it. 

### Miscellany
- Change interpolation methods in image transformation (in Caffe) from linear interpolation to cubic. 

Original caffe master branch and caffe-dev bracn use linear interpolation when they have to resize or transform `cv::Mat` images. By changing it to bicubic interpolation, I got several percents of improvement :) Please change it if you are using Caffe!  

### Final submissions
- Model averaging of 18 GoogLeNets (for NDSB) and 14 cxx\_net variants. 
1. For single model of model5 (cxx\_net variant) with offline augmentation produced about 74% accuracy(?) and 0.79 loss.
2. For single model of model5 (cxx\_net variant) with online augmentation produced about 0.78 loss.
3. For single model of model5 (cxx\_net variant) with online augmentation and multiple inference (8 inference) produced about 0.73 loss. 
4. Two models of 3 produced 0.71 loss
5. Four models of 3 produced 0.70 loss
5. 10 models of 3 produced 0.69 loss
6. For single model of GoogLeNet for NDSB produced about 75% accuracy and 0.80 loss. 
6. For single model of GoogLeNet for NDSB and a single model of model5 (cxx\_net variant) with online augmentation and multiple inference (8 inference) produced 0.68 loss. 
7. Model averaging of 18 GoogLeNets (for NDSB) and 14 cxx\_net variants produced 0.639206 loss. 
