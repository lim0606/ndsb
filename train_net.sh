#!/bin/bash

CAFFE_ROOT=/home/jaehyun/github/caffe-dev_tmp

$CAFFE_ROOT/build/tools/caffe train -solver model/solver.prototxt -gpu 0
