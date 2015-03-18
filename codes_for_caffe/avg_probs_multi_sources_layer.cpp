#include <algorithm>
#include <cfloat>
#include <vector>

#include "caffe/common.hpp"
#include "caffe/layer.hpp"
#include "caffe/syncedmem.hpp"
#include "caffe/util/math_functions.hpp"
#include "caffe/vision_layers.hpp"

namespace caffe {

using std::min;
using std::max;

template <typename Dtype>
void AVGProbsMultiSourcesLayer<Dtype>::LayerSetUp(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
  num_ = bottom[0]->num();
  channels_ = bottom[0]->channels();
  height_ = bottom[0]->height();
  width_ = bottom[0]->width();
  count_ = bottom[0]->count();

  CHECK_EQ(count_, num_*channels_*height_*width_);  

  avg_size_ = bottom.size(); 

  top[0]->Reshape(num_, channels_, height_, width_); 
}

template <typename Dtype>
void AVGProbsMultiSourcesLayer<Dtype>::Reshape(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
  num_ = bottom[0]->num(); 
  channels_ = bottom[0]->channels();
  height_ = bottom[0]->height();
  width_ = bottom[0]->width();
  count_ = bottom[0]->count(); 
 
  top[0]->Reshape(num_, channels_, height_, width_);
}

template <typename Dtype>
void AVGProbsMultiSourcesLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
  //const Dtype* bottom_data = bottom[0]->cpu_data();
  Dtype* top_data = top[0]->mutable_cpu_data();

  caffe_copy(count_, bottom[0]->cpu_data(), 
                 top_data); 
  caffe_scal(count_, Dtype(1. / avg_size_), top_data); 
  for (int b_id = 1; b_id < avg_size_; ++b_id) {
    caffe_axpy(count_, Dtype(1. / avg_size_), 
               bottom[b_id]->cpu_data(), 
               top_data);
  }   
  //check!
  //printf("%e", *(bottom[0]->cpu_data())); 
  //for (int i = 1; i < /*count_*/20; ++i) {
  //  printf(",%e", *(bottom[0]->cpu_data()+i));  
  //}
  //printf("\n\n"); 
  //printf("%e", *(top[0]->cpu_data())); 
  //for (int i = 1; i < /*count_*/20; ++i) {
  //  printf(",%e", *(top[0]->cpu_data()+i));  
  //}
  //printf("\n\n");
}

INSTANTIATE_CLASS(AVGProbsMultiSourcesLayer);
REGISTER_LAYER_CLASS(AVG_PROBS_MULTI_SOURCES, AVGProbsMultiSourcesLayer);
}  // namespace caffe
