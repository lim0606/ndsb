#!/bin/bash

CAFFE_ROOT=/home/jaehyun/github/caffe-dev_tmp

$CAFFE_ROOT/build/tools/caffe test -model model/train_val.prototxt -gpu 0 -iterations 500 -weights model/bvlc_reference_caffenet.caffemodel
