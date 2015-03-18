# Kaggle NDSB compettision with deep learning

I finished the competition with 22nd position (with small difference(?) from the winning solution)! 
I participated this competition to wrap-up and study the recent deep learning techniques, so I want to share my experiences.

## Kaggle NDSB competition
- See following the competition page :) 
 http://www.kaggle.com/c/datasciencebowl
- the blog post of the winning solution
 http://benanne.github.io/2015/03/17/plankton.html
 
## My approach
1. Data preprocessing and Data augmentation
2. Network architectures
- Variations of cxx_net
At first, I started from the model called cxx_net that are shared by Bing Xi (http://www.kaggle.com/users/43581/bing-xu). 
- see https://github.com/antinucleon/cxxnet/tree/master/example/kaggle_bowl

- GoogLeNet for NDSB

3. Training
- Batch normalization
- 
- 
4. Inference
- Multiple inference from single input
- Model averaging (from a single network architecture)
- Model averaging 

5. Miscellany
- Change interpolation methods in image transformation (in Caffe) from linear interpolation to cubic. 



My deep learning models

My final solution is a variation of the cxx_net and GoogleNet fitted to NDSB data.

