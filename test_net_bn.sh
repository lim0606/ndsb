#!/bin/bash

CAFFE_ROOT=/home/jaehyun/github/caffe-dev

$CAFFE_ROOT/build/tools/test_bn -train_model model/train_model5_bn.prototxt -test_model model/val_model5_bn.prototxt -weights snapshots/model5_bn_iter_132000.caffemodel -train_iterations 500 -gpu 0
