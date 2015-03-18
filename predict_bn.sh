#!/bin/bash

CAFFE_ROOT=/home/jaehyun/github/caffe-dev

$CAFFE_ROOT/build/tools/predict_bn -train_model model/train_googlenet_bn_var3_loss_avg.prototxt -test_model model/predict_googlenet_bn_var3_loss_avg.prototxt -weights snapshots/googlenet_bn_var3_iter_114000.caffemodel -train_iterations 1000 -labellist /media/data/kaggle/nationalDataScienceBowl/data/ndsb_labels.txt -outfile prediction_googlenet_bn_var3_loss_avg_avg8.114000.csv -gpu 0
