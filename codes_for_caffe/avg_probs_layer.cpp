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
void AVGProbsLayer<Dtype>::LayerSetUp(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
  num_ = bottom[0]->num();
  channels_ = bottom[0]->channels();
  height_ = bottom[0]->height();
  width_ = bottom[0]->width();
  K_ = channels_ * height_ * width_; 

  AVGProbsParameter avg_probs_param = this->layer_param_.avg_probs_param();
  multi_infer_size_ = avg_probs_param.multi_infer_size(); 

  CHECK_EQ(num_ % multi_infer_size_, 0); // num_images = multi_infer_size_ * batch_size

  batch_size_ = num_ / multi_infer_size_; 

  top[0]->Reshape(batch_size_, channels_, height_, width_); 
}

template <typename Dtype>
void AVGProbsLayer<Dtype>::Reshape(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
  num_ = bottom[0]->num(); 
  channels_ = bottom[0]->channels();
  height_ = bottom[0]->height();
  width_ = bottom[0]->width();
  K_ = channels_ * height_ * width_;
   
  CHECK_EQ(num_ , multi_infer_size_ * batch_size_); // num_images = multi_infer_size_ * batch_size_

  top[0]->Reshape(batch_size_, channels_, height_, width_);
}

template <typename Dtype>
void AVGProbsLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
  const Dtype* bottom_data = bottom[0]->cpu_data();
  Dtype* top_data = top[0]->mutable_cpu_data();

  for (int b_id = 0; b_id < batch_size_; ++b_id) {
    caffe_copy(K_, bottom_data + bottom[0]->offset(b_id * multi_infer_size_), 
                   top_data + top[0]->offset(b_id)); 
    caffe_scal(K_, Dtype(1. / multi_infer_size_), top_data + top[0]->offset(b_id)); 
    for (int m_id = 1; m_id < multi_infer_size_; ++m_id) {
      caffe_axpy(K_, Dtype(1. / multi_infer_size_), 
                 bottom_data + bottom[0]->offset(b_id * multi_infer_size_ + m_id), 
                 top_data + top[0]->offset(b_id));
    }   
    //check!
    //printf("%e", *(bottom_data + bottom[0]->offset(b_id * multi_infer_size_))); 
    //for (int i = 1; i < K_; ++i) {
    //  printf(",%e", *(bottom_data + bottom[0]->offset(b_id * multi_infer_size_) + i));  
    //}
    //printf("\n"); 
  }

}

INSTANTIATE_CLASS(AVGProbsLayer);
REGISTER_LAYER_CLASS(AVG_PROBS, AVGProbsLayer);
}  // namespace caffe
