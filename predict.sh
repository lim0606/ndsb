#!/bin/bash

CAFFE_ROOT=/home/jaehyun/github/caffe-dev

$CAFFE_ROOT/build/tools/predict test -model model/predict_model3.prototxt -gpu 0 -weights snapshots/model3_iter_174000.caffemodel -labellist /home/jaehyun/kaggle/nationalDataScienceBowl/data/ndsb_labels.txt -outfile prediction_model3.174000.csv
