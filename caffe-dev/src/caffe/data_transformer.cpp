#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>

#include <string>
#include <vector>

#include "caffe/data_transformer.hpp"
#include "caffe/util/io.hpp"
#include "caffe/util/math_functions.hpp"
#include "caffe/util/rng.hpp"

namespace caffe {

template<typename Dtype>
DataTransformer<Dtype>::DataTransformer(const TransformationParameter& param)
    : param_(param) {
  phase_ = Caffe::phase();
  // check if we want to use mean_file
  if (param_.has_mean_file()) {
    CHECK_EQ(param_.mean_value_size(), 0) <<
      "Cannot specify mean_file and mean_value at the same time";
    const string& mean_file = param.mean_file();
    LOG(INFO) << "Loading mean file from" << mean_file;
    BlobProto blob_proto;
    ReadProtoFromBinaryFileOrDie(mean_file.c_str(), &blob_proto);
    data_mean_.FromProto(blob_proto);
  }
  // check if we want to use mean_value
  if (param_.mean_value_size() > 0) {
    CHECK(param_.has_mean_file() == false) <<
      "Cannot specify mean_file and mean_value at the same time";
    for (int c = 0; c < param_.mean_value_size(); ++c) {
      mean_values_.push_back(param_.mean_value(c));
    }
  }
}

template<typename Dtype>
void DataTransformer<Dtype>::Transform(const Datum& datum,
                                       Dtype* transformed_data) {
  const string& data = datum.data();
  const int datum_channels = datum.channels();
  const int datum_height = datum.height();
  const int datum_width = datum.width();

  const int crop_size = param_.crop_size();
  const Dtype scale = param_.scale();
  const bool do_mirror = param_.mirror() && Rand(2);
  const bool has_mean_file = param_.has_mean_file();
  const bool has_uint8 = data.size() > 0;
  const bool has_mean_values = mean_values_.size() > 0;

  CHECK_GT(datum_channels, 0);
  CHECK_GE(datum_height, crop_size);
  CHECK_GE(datum_width, crop_size);

  Dtype* mean = NULL;
  if (has_mean_file) {
    CHECK_EQ(datum_channels, data_mean_.channels());
    CHECK_EQ(datum_height, data_mean_.height());
    CHECK_EQ(datum_width, data_mean_.width());
    mean = data_mean_.mutable_cpu_data();
  }
  if (has_mean_values) {
    CHECK(mean_values_.size() == 1 || mean_values_.size() == datum_channels) <<
     "Specify either 1 mean_value or as many as channels: " << datum_channels;
    if (datum_channels > 1 && mean_values_.size() == 1) {
      // Replicate the mean_value for simplicity
      for (int c = 1; c < datum_channels; ++c) {
        mean_values_.push_back(mean_values_[0]);
      }
    }
  }

  int height = datum_height;
  int width = datum_width;

  int h_off = 0;
  int w_off = 0;
  if (crop_size) {
    height = crop_size;
    width = crop_size;
    // We only do random crop when we do training.
    if (phase_ == Caffe::TRAIN) {
      h_off = Rand(datum_height - crop_size + 1);
      w_off = Rand(datum_width - crop_size + 1);
    } else {
      h_off = (datum_height - crop_size) / 2;
      w_off = (datum_width - crop_size) / 2;
    }
  }

  Dtype datum_element;
  int top_index, data_index;
  for (int c = 0; c < datum_channels; ++c) {
    for (int h = 0; h < height; ++h) {
      for (int w = 0; w < width; ++w) {
        data_index = (c * datum_height + h_off + h) * datum_width + w_off + w;
        if (do_mirror) {
          top_index = (c * height + h) * width + (width - 1 - w);
        } else {
          top_index = (c * height + h) * width + w;
        }
        if (has_uint8) {
          datum_element =
            static_cast<Dtype>(static_cast<uint8_t>(data[data_index]));
        } else {
          datum_element = datum.float_data(data_index);
        }
        if (has_mean_file) {
          transformed_data[top_index] =
            (datum_element - mean[data_index]) * scale;
        } else {
          if (has_mean_values) {
            transformed_data[top_index] =
              (datum_element - mean_values_[c]) * scale;
          } else {
            transformed_data[top_index] = datum_element * scale;
          }
        }
      }
    }
  }
}

// jhlim
template<typename Dtype>
void DataTransformer<Dtype>::Transform(const Datum& datum,
                                       Blob<Dtype>* transformed_blob, int batch_id, int image_id) {
  cv::Mat cv_img = DatumToCVMat(datum); 
  char filename[1024]; // The filename.
  // Put "file" then k then ".txt" in to filename.
  snprintf(filename, sizeof(char) * 1024, 
      "/media/data/kaggle/nationalDataScienceBowl/models/caffe_data_aug/tmp/%d_%d_img_lmdb.jpg", batch_id, image_id);
  cv::imwrite(filename, cv_img); 

  Transform(datum, transformed_blob); 
}

template<typename Dtype>
void DataTransformer<Dtype>::Transform(const Datum& datum,
                                       Blob<Dtype>* transformed_blob) {
  const int datum_channels = datum.channels();
  const int datum_height = datum.height();
  const int datum_width = datum.width();

  const int channels = transformed_blob->channels();
  const int height = transformed_blob->height();
  const int width = transformed_blob->width();
  const int num = transformed_blob->num();

  CHECK_EQ(channels, datum_channels);
  CHECK_LE(height, datum_height);
  CHECK_LE(width, datum_width);
  CHECK_GE(num, 1);

  const int crop_size = param_.crop_size();

  if (crop_size) {
    CHECK_EQ(crop_size, height);
    CHECK_EQ(crop_size, width);
  } else {
    CHECK_EQ(datum_height, height);
    CHECK_EQ(datum_width, width);
  }

  Dtype* transformed_data = transformed_blob->mutable_cpu_data();
  Transform(datum, transformed_data);
}

template<typename Dtype>
void DataTransformer<Dtype>::Transform(const vector<Datum> & datum_vector,
                                       Blob<Dtype>* transformed_blob) {
  const int datum_num = datum_vector.size();
  const int num = transformed_blob->num();
  const int channels = transformed_blob->channels();
  const int height = transformed_blob->height();
  const int width = transformed_blob->width();

  CHECK_GT(datum_num, 0) << "There is no datum to add";
  CHECK_LE(datum_num, num) <<
    "The size of datum_vector must be smaller than transformed_blob->num()";
  Blob<Dtype> uni_blob(1, channels, height, width);
  for (int item_id = 0; item_id < datum_num; ++item_id) {
    int offset = transformed_blob->offset(item_id);
    uni_blob.set_cpu_data(transformed_blob->mutable_cpu_data() + offset);
    Transform(datum_vector[item_id], &uni_blob);
  }
}

// jhlim
template<typename Dtype>
void DataTransformer<Dtype>::TransformNDSB(const cv::Mat& cv_img,
                                       Blob<Dtype>* transformed_blob) {
  const int img_channels = cv_img.channels(); // channel should be 1
  /*const*/ int img_height = cv_img.rows;
  /*const*/ int img_width = cv_img.cols;

  const int channels = transformed_blob->channels();
  const int height = transformed_blob->height();
  const int width = transformed_blob->width();
  const int num = transformed_blob->num();

  CHECK_EQ(channels, img_channels);
//  CHECK_LE(height, img_height);
//  CHECK_LE(width, img_width);
  CHECK_GE(num, 1);
//  printf("img_height: %d", img_height);
//  printf(", img_width: %d", img_width); 
//  printf(", height: %d", height); 
//  printf(", width: %d", width); 
//  printf("\n"); 

  CHECK(cv_img.depth() == CV_8U) << "Image data type must be unsigned byte";

  const int crop_size = param_.crop_size();
  const Dtype scale = param_.scale();
  const bool do_mirror = param_.mirror() && Rand(2);
  const bool has_mean_file = param_.has_mean_file();
  const bool has_mean_values = mean_values_.size() > 0;

  CHECK_GT(img_channels, 0);
//  CHECK_GE(img_height, crop_size);
//  CHECK_GE(img_width, crop_size);
  //printf("crop_size: %d\n", crop_size); 

  Dtype* mean = NULL;
  if (has_mean_file) {
    CHECK_EQ(img_channels, data_mean_.channels());
    CHECK_EQ(img_height, data_mean_.height());
    CHECK_EQ(img_width, data_mean_.width());
    mean = data_mean_.mutable_cpu_data();
  }
  if (has_mean_values) {
    CHECK(mean_values_.size() == 1 || mean_values_.size() == img_channels) <<
     "Specify either 1 mean_value or as many as channels: " << img_channels;
    if (img_channels > 1 && mean_values_.size() == 1) {
      // Replicate the mean_value for simplicity
      for (int c = 1; c < img_channels; ++c) {
        mean_values_.push_back(mean_values_[0]);
      }
    }
  }
 
  //printf("asdfasdfasdfasdf1\n"); 
  ////// jhlim
  // initialize transformation parameter 
  float c00=1;
  float c01=0;
  float c10=0;
  float c11=1;
  float backgroundColor = 0;
  float tmp = 0;

  caffe_rng_uniform<float>(1,-0.2,0.2,&tmp);
  c00*=1+tmp; // x stretch
  caffe_rng_uniform<float>(1,-0.2,0.2,&tmp);
  c11*=1+tmp; // y stretch
  
  if (phase_ == Caffe::TRAIN) {
    if (Rand(2) == 0) c00*=-1; // Horizontal flip
 
    int r=Rand(3);

    float alpha = 0; caffe_rng_uniform<float>(1,-0.2,0.2,&alpha); 
    if (r==0) { c01 += alpha*c00; c11+=alpha*c10; }
    if (r==0) { c10 += alpha*c00; c11+=alpha*c01; }
    if (r==2) {
      float c=cos(alpha); float s=sin(alpha); 
      float t00=c00*c-c01*s; float t01=c00*s+c01*c; c00=t00; c01=t01;
      float t10=c10*c-c11*s; float t11=c10*s+c11*c; c10=t10; c11=t11;
    }
  }

  //printf("asdfasdfasdfasdf2\n"); 
  // do affine transformation
  cv::Mat warp, dst;
  cv::Point2f srcTri[3];
  cv::Point2f dstTri[3];
 
  int X=img_width-1;
  int Y=img_height-1;
  int x=0;
  int y=0;
  srcTri[0]=cv::Point2f(x,y);
  srcTri[1]=cv::Point2f(X,y);
  srcTri[2]=cv::Point2f(x,Y);
  dstTri[0]=cv::Point2f(x*c00+y*c10,x*c01+y*c11);
  dstTri[1]=cv::Point2f(X*c00+y*c10,X*c01+y*c11);
  dstTri[2]=cv::Point2f(x*c00+Y*c10,x*c01+Y*c11);
  float m;
  m=std::min(std::min(std::min(dstTri[0].x,dstTri[1].x),dstTri[2].x),dstTri[1].x+dstTri[2].x);
  dstTri[0].x-=m;
  dstTri[1].x-=m;
  dstTri[2].x-=m;
  m=std::min(std::min(std::min(dstTri[0].y,dstTri[1].y),dstTri[2].y),dstTri[1].y+dstTri[2].y);
  dstTri[0].y-=m;
  dstTri[1].y-=m;
  dstTri[2].y-=m;
  dst = cv::Mat::zeros(std::max(std::max(std::max(dstTri[0].y,dstTri[1].y),dstTri[2].y),dstTri[1].y+dstTri[2].y),
                       std::max(std::max(std::max(dstTri[0].x,dstTri[1].x),dstTri[2].x),dstTri[1].x+dstTri[2].x),
                       cv_img.type());
  warp = cv::getAffineTransform( srcTri, dstTri );
  cv::warpAffine( cv_img, dst, warp, dst.size(), cv::INTER_CUBIC, cv::BORDER_CONSTANT, cv::Scalar::all(backgroundColor));
  ////// jhlim until here
  
  img_height = dst.rows; // update img_height and width with affine image
  img_width = dst.cols;
  //printf("img_height: %d, img_width: %d\n", img_height, img_width);
  if (img_height > img_width) {
    int margin_left = (int)((float)(img_height-img_width)/2.);
    int margin_right = (img_height-img_width) - margin_left; 
    copyMakeBorder(dst, dst, 0, 0, margin_left, margin_right, cv::BORDER_CONSTANT, cv::Scalar::all(backgroundColor)); 
  } else if (img_height < img_width) {
    int margin_up = (int)((float)(img_width-img_height)/2.);
    int margin_down = (img_width-img_height) - margin_up; 
    copyMakeBorder(dst, dst, margin_up, margin_down, 0, 0, cv::BORDER_CONSTANT, cv::Scalar::all(backgroundColor)); 

  }
  img_height = dst.rows; // update img_height and width with affine image
  img_width = dst.cols;
 
//  printf("(final) img_height: %d, img_width: %d\n", img_height, img_width);

//  printf("asdfasdfasdfasdf3\n"); 
  // do crop
  int h_off = 0;
  int w_off = 0;
  // cv::Mat cv_cropped_img = cv_img;
  cv::Mat cv_cropped_img(width, height, CV_8UC(channels), cv::Scalar::all(backgroundColor));
  // cv_cropped_img.create(width, height, CV_8UC(channels)); 
/*  if (crop_size) {
    CHECK_EQ(crop_size, height);
    CHECK_EQ(crop_size, width);
    // We only do random crop when we do training.
    if (phase_ == Caffe::TRAIN) {
      h_off = Rand(img_height - crop_size + 1);
      w_off = Rand(img_width - crop_size + 1);
    } else {
      h_off = (img_height - crop_size) / 2;
      w_off = (img_width - crop_size) / 2;
    }
    cv::Rect roi(w_off, h_off, crop_size, crop_size);
    cv_cropped_img = cv_img(roi);
  } else {
    CHECK_EQ(img_height, height);
    CHECK_EQ(img_width, width);
  } */
  if (crop_size < img_height) {
    //printf("asdfasdfasdfasdf3.5\n");
    //printf("crop_size: %d, img_height: %d \n", crop_size, img_height); 
    CHECK_EQ(crop_size, height);
    CHECK_EQ(crop_size, width);
    // We only do random crop when we do training.
    if (phase_ == Caffe::TRAIN) {
      h_off = Rand(img_height - crop_size + 1);
      w_off = Rand(img_width - crop_size + 1);
    } else {
      h_off = (img_height - crop_size) / 2;
      w_off = (img_width - crop_size) / 2;
    }
    cv::Rect roi(w_off, h_off, crop_size, crop_size);
    //cv_cropped_img = cv_img(roi);
    cv_cropped_img = dst(roi);
  } else if (crop_size > img_height) { // jhlim
    //printf("asdfasdfasdfasdf4\n"); 
    // do resizing and random translation
    float maxTranslation = 16;
    int xOffset=-img_width/2; // 32
    int yOffset=-img_height/2; // 32
    //if (phase_ == Caffe::TRAIN) {
      xOffset+=Rand(maxTranslation*2+1)-maxTranslation;
      yOffset+=Rand(maxTranslation*2+1)-maxTranslation;
    //}
    // replace a part of image with another 
    cv::Mat roi(cv_cropped_img, cv::Rect(std::max(xOffset + width/2, 0), 
                                       std::max(yOffset + height/2, 0),
                                       std::min(img_width, width-xOffset-width/2),
                                       std::min(img_height, height-yOffset-height/2)));

//    if(xOffset + width/2 >= 0 && xOffset + width/2 + img_width <= width && 
//       yOffset + height/2 >= 0 && yOffset + height/2 + img_height <= height) {
//      dst.copyTo(roi); 
//    } else {
      float rx = abs(std::min((int)(xOffset + width/2), 0));
      float ry = abs(std::min((int)(yOffset + height/2), 0));
      float rw = img_width-std::max(xOffset+width/2+img_width-width,0) - rx;
      float rh = img_height-std::max(yOffset+height/2+img_height-height,0) - ry;
      cv::Mat dst_roi(dst, cv::Rect(rx,ry,rw,rh));
      dst_roi.copyTo(roi);
//    }
      
      dst_roi.release(); 

//    cv::imwrite("affine.png", cv_cropped_img); 
//    cv::imwrite("original.png", cv_img); 
    
  } else {
    CHECK_EQ(img_height, height);
    CHECK_EQ(img_width, width);
  } 

  warp.release(); 
  dst.release(); 

  //printf("asdfasdfasdfasdf5\n"); 
  // do re-scaling from 0-255 to 0-1  (jhlim)
  cv_cropped_img.convertTo(cv_cropped_img, CV_32FC(channels)); 
  float div = std::max(255-backgroundColor, backgroundColor);
  cv_cropped_img -= div;
  cv_cropped_img *= 1/div;
  // re-scaling until here

  CHECK(cv_cropped_img.data);

  Dtype* transformed_data = transformed_blob->mutable_cpu_data();
  int top_index;
  for (int h = 0; h < height; ++h) {
    const uchar* ptr = cv_cropped_img.ptr<uchar>(h);
    int img_index = 0;
    for (int w = 0; w < width; ++w) {
      for (int c = 0; c < img_channels; ++c) {
        if (do_mirror) {
          top_index = (c * height + h) * width + (width - 1 - w);
        } else {
          top_index = (c * height + h) * width + w;
        }
        // int top_index = (c * height + h) * width + w;
        Dtype pixel = static_cast<Dtype>(ptr[img_index++]);
        if (has_mean_file) {
          int mean_index = (c * img_height + h_off + h) * img_width + w_off + w;
          transformed_data[top_index] =
            (pixel - mean[mean_index]) * scale;
        } else {
          if (has_mean_values) {
            transformed_data[top_index] =
              (pixel - mean_values_[c]) * scale;
          } else {
            transformed_data[top_index] = pixel * scale;
          }
        }
      }
    }
  }
}

// jhlim
template<typename Dtype>
void DataTransformer<Dtype>::TransformAffine(const cv::Mat& cv_img,
                                       Blob<Dtype>* transformed_blob) {
  const int img_channels = cv_img.channels();
  /*const*/ int img_height = cv_img.rows;
  /*const*/ int img_width = cv_img.cols;

  const int channels = transformed_blob->channels();
  const int height = transformed_blob->height();
  const int width = transformed_blob->width();
  const int num = transformed_blob->num();

  CHECK_EQ(channels, img_channels);
//  CHECK_LE(height, img_height);
//  CHECK_LE(width, img_width);
  CHECK_GE(num, 1);

  CHECK(cv_img.depth() == CV_8U) << "Image data type must be unsigned byte";

  const int crop_size = param_.crop_size();
  const Dtype scale = param_.scale();
  const bool do_mirror = param_.mirror() && Rand(2);
  const bool has_mean_file = param_.has_mean_file();
  const bool has_mean_values = mean_values_.size() > 0;

  CHECK_GT(img_channels, 0);
//  CHECK_GE(img_height, crop_size);
//  CHECK_GE(img_width, crop_size);

  Dtype* mean = NULL;
  if (has_mean_file) {
    CHECK_EQ(img_channels, data_mean_.channels());
    CHECK_EQ(img_height, data_mean_.height());
    CHECK_EQ(img_width, data_mean_.width());
    mean = data_mean_.mutable_cpu_data();
  }
  if (has_mean_values) {
    CHECK(mean_values_.size() == 1 || mean_values_.size() == img_channels) <<
     "Specify either 1 mean_value or as many as channels: " << img_channels;
    if (img_channels > 1 && mean_values_.size() == 1) {
      // Replicate the mean_value for simplicity
      for (int c = 1; c < img_channels; ++c) {
        mean_values_.push_back(mean_values_[0]);
      }
    }
  }
 
  //printf("asdfasdfasdfasdf1\n"); 
  ////// jhlim
  // initialize transformation parameter 
  float c00=1;
  float c01=0;
  float c10=0;
  float c11=1;
  float backgroundColor = 128;
  float tmp = 0;

  caffe_rng_uniform<float>(1,-0.2,0.2,&tmp);
  c00*=1+tmp; // x stretch
  caffe_rng_uniform<float>(1,-0.2,0.2,&tmp);
  c11*=1+tmp; // y stretch
  
//  if (phase_ == Caffe::TRAIN) {
    if (Rand(2) == 0) c00*=-1; // Horizontal flip
 
    int r=Rand(3);

    float alpha = 0; caffe_rng_uniform<float>(1,-0.2,0.2,&alpha); 
    if (r==0) { c01 += alpha*c00; c11+=alpha*c10; }
    if (r==0) { c10 += alpha*c00; c11+=alpha*c01; }
    if (r==2) {
      float c=cos(alpha); float s=sin(alpha); 
      float t00=c00*c-c01*s; float t01=c00*s+c01*c; c00=t00; c01=t01;
      float t10=c10*c-c11*s; float t11=c10*s+c11*c; c10=t10; c11=t11;
    }
//  }

  //printf("asdfasdfasdfasdf2\n"); 
  // do affine transformation
  cv::Mat warp, dst;
  cv::Point2f srcTri[3];
  cv::Point2f dstTri[3];
 
  int X=img_width-1;
  int Y=img_height-1;
  int x=0;
  int y=0;
  srcTri[0]=cv::Point2f(x,y);
  srcTri[1]=cv::Point2f(X,y);
  srcTri[2]=cv::Point2f(x,Y);
  dstTri[0]=cv::Point2f(x*c00+y*c10,x*c01+y*c11);
  dstTri[1]=cv::Point2f(X*c00+y*c10,X*c01+y*c11);
  dstTri[2]=cv::Point2f(x*c00+Y*c10,x*c01+Y*c11);
  float m;
  m=std::min(std::min(std::min(dstTri[0].x,dstTri[1].x),dstTri[2].x),dstTri[1].x+dstTri[2].x);
  dstTri[0].x-=m;
  dstTri[1].x-=m;
  dstTri[2].x-=m;
  m=std::min(std::min(std::min(dstTri[0].y,dstTri[1].y),dstTri[2].y),dstTri[1].y+dstTri[2].y);
  dstTri[0].y-=m;
  dstTri[1].y-=m;
  dstTri[2].y-=m;
  dst = cv::Mat::zeros(std::max(std::max(std::max(dstTri[0].y,dstTri[1].y),dstTri[2].y),dstTri[1].y+dstTri[2].y),
                       std::max(std::max(std::max(dstTri[0].x,dstTri[1].x),dstTri[2].x),dstTri[1].x+dstTri[2].x),
                       cv_img.type());
  warp = cv::getAffineTransform( srcTri, dstTri );
  cv::warpAffine( cv_img, dst, warp, dst.size(),cv::INTER_CUBIC, cv::BORDER_CONSTANT, cv::Scalar(backgroundColor,backgroundColor,backgroundColor));
  ////// jhlim until here
  
  img_height = dst.rows; // update img_height and width with affine image
  img_width = dst.cols;
  //printf("img_height: %d, img_width: %d\n", img_height, img_width);
  if (img_height > img_width) {
    int margin_left = (int)((float)(img_height-img_width)/2.);
    int margin_right = (img_height-img_width) - margin_left; 
    copyMakeBorder(dst, dst, 0, 0, margin_left, margin_right, cv::BORDER_CONSTANT, cv::Scalar::all(backgroundColor)); 
  } else if (img_height < img_width) {
    int margin_up = (int)((float)(img_width-img_height)/2.);
    int margin_down = (img_width-img_height) - margin_up; 
    copyMakeBorder(dst, dst, margin_up, margin_down, 0, 0, cv::BORDER_CONSTANT, cv::Scalar::all(backgroundColor)); 

  }
  img_height = dst.rows; // update img_height and width with affine image
  img_width = dst.cols;
 
//  printf("(final) img_height: %d, img_width: %d\n", img_height, img_width);


  //printf("asdfasdfasdfasdf3\n"); 
  // do crop
  int h_off = 0;
  int w_off = 0;
  // cv::Mat cv_cropped_img = cv_img;
  cv::Mat cv_cropped_img(width, height, CV_8UC(channels), cv::Scalar::all(backgroundColor));
  // cv_cropped_img.create(width, height, CV_8UC(channels)); 
/*  if (crop_size) {
    CHECK_EQ(crop_size, height);
    CHECK_EQ(crop_size, width);
    // We only do random crop when we do training.
    if (phase_ == Caffe::TRAIN) {
      h_off = Rand(img_height - crop_size + 1);
      w_off = Rand(img_width - crop_size + 1);
    } else {
      h_off = (img_height - crop_size) / 2;
      w_off = (img_width - crop_size) / 2;
    }
    cv::Rect roi(w_off, h_off, crop_size, crop_size);
    cv_cropped_img = cv_img(roi);
  } else {
    CHECK_EQ(img_height, height);
    CHECK_EQ(img_width, width);
  } */
  if (crop_size < img_height) {
    CHECK_EQ(crop_size, height);
    CHECK_EQ(crop_size, width);
    // We only do random crop when we do training.
    if (phase_ == Caffe::TRAIN) {
      h_off = Rand(img_height - crop_size + 1);
      w_off = Rand(img_width - crop_size + 1);
    } else {
      h_off = (img_height - crop_size) / 2;
      w_off = (img_width - crop_size) / 2;
    }
    cv::Rect roi(w_off, h_off, crop_size, crop_size);
    //cv_cropped_img = cv_img(roi);
    cv_cropped_img = dst(roi);
  } else if (crop_size > img_height) { // jhlim
    //printf("asdfasdfasdfasdf4\n"); 
    // do resizing and random translation
    float maxTranslation = 16;
    int xOffset=-img_width/2; // 32
    int yOffset=-img_height/2; // 32
    //if (phase_ == Caffe::TRAIN) {
      xOffset+=Rand(maxTranslation*2+1)-maxTranslation;
      yOffset+=Rand(maxTranslation*2+1)-maxTranslation;
    //}
    // replace a part of image with another 
    cv::Mat roi(cv_cropped_img, cv::Rect(std::max(xOffset + width/2, 0), 
                                       std::max(yOffset + height/2, 0),
                                       std::min(img_width, width-xOffset-width/2),
                                       std::min(img_height, height-yOffset-height/2)));

//    if(xOffset + width/2 >= 0 && xOffset + width/2 + img_width <= width && 
//       yOffset + height/2 >= 0 && yOffset + height/2 + img_height <= height) {
//      dst.copyTo(roi); 
//    } else {
      float rx = abs(std::min((int)(xOffset + width/2), 0));
      float ry = abs(std::min((int)(yOffset + height/2), 0));
      float rw = img_width-std::max(xOffset+width/2+img_width-width,0) - rx;
      float rh = img_height-std::max(yOffset+height/2+img_height-height,0) - ry;
      cv::Mat dst_roi(dst, cv::Rect(rx,ry,rw,rh));
      dst_roi.copyTo(roi);
//    }
      
      dst_roi.release(); 

//    cv::imwrite("affine.png", cv_cropped_img); 
//    cv::imwrite("original.png", cv_img); 
    
  } else {
    CHECK_EQ(img_height, height);
    CHECK_EQ(img_width, width);
  } 
  
  warp.release(); 
  dst.release(); 

 //printf("asdfasdfasdfasdf5\n"); 
  // do re-scaling from 0-255 to 0-1  (jhlim)
  cv_cropped_img.convertTo(cv_cropped_img, CV_32FC(channels)); 
  float div = std::max(255-backgroundColor, backgroundColor);
  cv_cropped_img -= div;
  cv_cropped_img *= 1/div;
  // re-scaling until here
  
  //printf("asdfasdfasdfasdf6\n"); 
  // do color shift (jhlim)
  float sigma1=0.2;
  float sigma2=0.3;
  float sigma3=0.6;
  float sigma4=0.6; 
  float tmp2=0;
  
  vector<float> delta1(channels);
  vector<float> delta2(channels);
  vector<float> delta3(channels);
  vector<float> delta4(channels);
  for (int j=0;j<channels;j++) {
    caffe_rng_gaussian<float>(1,0,sigma1,&tmp2);
    delta1[j]=tmp2;
    caffe_rng_gaussian<float>(1,0,sigma2,&tmp2);
    delta2[j]=tmp2;
    caffe_rng_gaussian<float>(1,0,sigma3,&tmp2);
    delta3[j]=tmp2; 
    caffe_rng_gaussian<float>(1,0,sigma4,&tmp2);
    delta4[j]=tmp2;
  }

  CHECK_EQ(channels, 3); // current implementation only support 3 channel image input
  for (int x=0;x<width;x++) {
    for (int y=0;y<height;y++) {
      if (cv_cropped_img.at<cv::Vec3f>(y,x)[0] != 0 &&
          cv_cropped_img.at<cv::Vec3f>(y,x)[1] != 0 &&
          cv_cropped_img.at<cv::Vec3f>(y,x)[2] != 0) {
        for (int j=0;j<channels;j++) {
          //int k=y*input.spatialSize+x;
          //if (output.grids[i][k]!=bg) {
          //  output.features.hVector()[output.grids[i][k]*output.nFeatures+j]+=
            cv_cropped_img.at<cv::Vec3f>(y,x)[j] +=
            delta1[j]+
            delta2[j]*cos(cv_cropped_img.at<cv::Vec3f>(y,x)[j]*3.1415926535/2)+
            delta3[j]*(x*1.0f/width-0.5f)+
            delta4[j]*(y*1.0f/height-0.5f);
          //}
        }
      }
    }
  }
  // color shift until here

  CHECK(cv_cropped_img.data);

  Dtype* transformed_data = transformed_blob->mutable_cpu_data();
  int top_index;
  for (int h = 0; h < height; ++h) {
    const uchar* ptr = cv_cropped_img.ptr<uchar>(h);
    int img_index = 0;
    for (int w = 0; w < width; ++w) {
      for (int c = 0; c < img_channels; ++c) {
        if (do_mirror) {
          top_index = (c * height + h) * width + (width - 1 - w);
        } else {
          top_index = (c * height + h) * width + w;
        }
        // int top_index = (c * height + h) * width + w;
        Dtype pixel = static_cast<Dtype>(ptr[img_index++]);
        if (has_mean_file) {
          int mean_index = (c * img_height + h_off + h) * img_width + w_off + w;
          transformed_data[top_index] =
            (pixel - mean[mean_index]) * scale;
        } else {
          if (has_mean_values) {
            transformed_data[top_index] =
              (pixel - mean_values_[c]) * scale;
          } else {
            transformed_data[top_index] = pixel * scale;
          }
        }
      }
    }
  }
}

template<typename Dtype>
void DataTransformer<Dtype>::Transform(const cv::Mat& cv_img,
                                       Blob<Dtype>* transformed_blob) {
  const int img_channels = cv_img.channels();
  const int img_height = cv_img.rows;
  const int img_width = cv_img.cols;

  const int channels = transformed_blob->channels();
  const int height = transformed_blob->height();
  const int width = transformed_blob->width();
  const int num = transformed_blob->num();

  CHECK_EQ(channels, img_channels);
  CHECK_LE(height, img_height);
  CHECK_LE(width, img_width);
  CHECK_GE(num, 1);

  CHECK(cv_img.depth() == CV_8U) << "Image data type must be unsigned byte";

  const int crop_size = param_.crop_size();
  const Dtype scale = param_.scale();
  const bool do_mirror = param_.mirror() && Rand(2);
  const bool has_mean_file = param_.has_mean_file();
  const bool has_mean_values = mean_values_.size() > 0;

  CHECK_GT(img_channels, 0);
  CHECK_GE(img_height, crop_size);
  CHECK_GE(img_width, crop_size);

  Dtype* mean = NULL;
  if (has_mean_file) {
    CHECK_EQ(img_channels, data_mean_.channels());
    CHECK_EQ(img_height, data_mean_.height());
    CHECK_EQ(img_width, data_mean_.width());
    mean = data_mean_.mutable_cpu_data();
  }
  if (has_mean_values) {
    CHECK(mean_values_.size() == 1 || mean_values_.size() == img_channels) <<
     "Specify either 1 mean_value or as many as channels: " << img_channels;
    if (img_channels > 1 && mean_values_.size() == 1) {
      // Replicate the mean_value for simplicity
      for (int c = 1; c < img_channels; ++c) {
        mean_values_.push_back(mean_values_[0]);
      }
    }
  }

  int h_off = 0;
  int w_off = 0;
  cv::Mat cv_cropped_img = cv_img;
  if (crop_size) {
    CHECK_EQ(crop_size, height);
    CHECK_EQ(crop_size, width);
    // We only do random crop when we do training.
    if (phase_ == Caffe::TRAIN) {
      h_off = Rand(img_height - crop_size + 1);
      w_off = Rand(img_width - crop_size + 1);
    } else {
      h_off = (img_height - crop_size) / 2;
      w_off = (img_width - crop_size) / 2;
    }
    cv::Rect roi(w_off, h_off, crop_size, crop_size);
    cv_cropped_img = cv_img(roi);
  } else {
    CHECK_EQ(img_height, height);
    CHECK_EQ(img_width, width);
  }

  CHECK(cv_cropped_img.data);

  Dtype* transformed_data = transformed_blob->mutable_cpu_data();
  int top_index;
  for (int h = 0; h < height; ++h) {
    const uchar* ptr = cv_cropped_img.ptr<uchar>(h);
    int img_index = 0;
    for (int w = 0; w < width; ++w) {
      for (int c = 0; c < img_channels; ++c) {
        if (do_mirror) {
          top_index = (c * height + h) * width + (width - 1 - w);
        } else {
          top_index = (c * height + h) * width + w;
        }
        // int top_index = (c * height + h) * width + w;
        Dtype pixel = static_cast<Dtype>(ptr[img_index++]);
        if (has_mean_file) {
          int mean_index = (c * img_height + h_off + h) * img_width + w_off + w;
          transformed_data[top_index] =
            (pixel - mean[mean_index]) * scale;
        } else {
          if (has_mean_values) {
            transformed_data[top_index] =
              (pixel - mean_values_[c]) * scale;
          } else {
            transformed_data[top_index] = pixel * scale;
          }
        }
      }
    }
  }
}

// jhlim this function is only for the ImageDataAffineLayer
cv::Point2f rotPoint(const cv::Mat &R, const cv::Point2f &p) {
    cv::Point2f rp;
    rp.x = (float)(R.at<double>(0,0)*p.x + R.at<double>(0,1)*p.y + R.at<double>(0,2));
    rp.y = (float)(R.at<double>(1,0)*p.x + R.at<double>(1,1)*p.y + R.at<double>(1,2));
    return rp;
}
cv::Size rotatedImageSize(const cv::Mat &R, const cv::Rect &bb) {
    //Rotate the rectangle coordinates
    vector<cv::Point2f> rp;
    rp.push_back(rotPoint(R,cv::Point2f(bb.x,bb.y)));
    rp.push_back(rotPoint(R,cv::Point2f(bb.x + bb.width,bb.y)));
    rp.push_back(rotPoint(R,cv::Point2f(bb.x + bb.width,bb.y+bb.height)));
    rp.push_back(rotPoint(R,cv::Point2f(bb.x,bb.y+bb.height)));
    //Find float bounding box r
    float x = rp[0].x;
    float y = rp[0].y;
    float left = x, right = x, up = y, down = y;
    for(int i = 1; i<4; ++i)
    {
        x = rp[i].x;
        y = rp[i].y;
        if(left > x) left = x;
        if(right < x) right = x;
        if(up > y) up = y;
        if(down < y) down = y;
    }
    int w = (int)(right - left + 0.5);
    int h = (int)(down - up + 0.5);
    return cv::Size(w,h);
}

template<typename Dtype>
void DataTransformer<Dtype>::TransformAffine(const cv::Mat& cv_img_origin, 
                                       int new_height, int new_width,
                                       Blob<Dtype>* transformed_blob) {
  CHECK_GT(new_height, 0); 
  CHECK_GT(new_width, 0); 
  
/*  cv::Mat cv_img_tmp; 
  cv::resize(cv_img_origin, cv_img_tmp, cv::Size(new_width, new_height), 0, 0, cv::INTER_CUBIC);
  cv::imwrite("/media/data/kaggle/nationalDataScienceBowl/models/caffe_data_aug/img_origin.jpg", cv_img_tmp); */

  const int img_channels = cv_img_origin.channels();
  const int img_height = cv_img_origin.rows;
  const int img_width = cv_img_origin.cols;

  const int channels = transformed_blob->channels();
  const int height = transformed_blob->height();
  const int width = transformed_blob->width();
  const int num = transformed_blob->num();

  CHECK_LE(height, new_height); // CHECK_LE(height, img_height);
  CHECK_LE(width, new_width); // CHECK_LE(width, img_width);
  CHECK_EQ(channels, img_channels);
  CHECK_GE(num, 1);

  CHECK(cv_img_origin.depth() == CV_8U) << "Image data type must be unsigned byte";

  const int crop_size = param_.crop_size();
  const Dtype scale = param_.scale();
  const bool do_mirror = param_.mirror() && Rand(2);
  const bool has_mean_file = param_.has_mean_file();
  const bool has_mean_values = mean_values_.size() > 0;

  CHECK_GT(img_channels, 0);
  CHECK_GE(new_height, crop_size); // CHECK_GE(img_height, crop_size); 
  CHECK_GE(new_width, crop_size); // CHECK_GE(img_width, crop_size);

  // make transform matrix
  cv::Mat cv_img_rot, cv_img_transformed;  
  // initialize transformation parameter
  float backgroundColor = param_.affine_param().background_color();
  // rotation first
  float angle = 0;
  caffe_rng_uniform<float>(1,
                           -param_.affine_param().rotation_unirnd_range(),
                           param_.affine_param().rotation_unirnd_range(),
                           &angle);
  //printf("angle: %.2f\n", angle); 
  cv::Mat rot = cv::getRotationMatrix2D(cv::Point2f(img_width/2.f, img_height/2.f), angle, 1.0); 
  cv::Size rs = rotatedImageSize(rot, cv::Rect(0,0, img_width, img_height));
  cv_img_rot.create(rs,cv_img_origin.type());
  cv::warpAffine( cv_img_origin, cv_img_rot, rot, cv_img_transformed.size(), cv::INTER_CUBIC, cv::BORDER_CONSTANT, cv::Scalar::all(backgroundColor));
  // stretch
  float tmp = 0;
  caffe_rng_uniform<float>(1,
                           -param_.affine_param().x_stretch_unirnd_range(),
                           param_.affine_param().x_stretch_unirnd_range(),
                           &tmp);
  float x_stretch=1+tmp; // x stretch
  caffe_rng_uniform<float>(1,
                           -param_.affine_param().y_stretch_unirnd_range(),
                           param_.affine_param().y_stretch_unirnd_range(),
                           &tmp);
  float y_stretch=1+tmp; // y stretch
  cv::resize(cv_img_rot, cv_img_transformed, 
             cv::Size(cv_img_rot.cols * x_stretch, cv_img_rot.rows * y_stretch), 0, 0, cv::INTER_CUBIC);

  ////// affine until here

  // do jiggle
  cv::Mat cv_img_transformed_bak; 
  if (param_.affine_param().has_max_translation()) {
    cv_img_transformed_bak = cv_img_transformed;

    int h_off_jiggle = 0; 
    int w_off_jiggle = 0; 

    if (param_.affine_param().max_translation() > 0) {
      if (cv_img_transformed.rows * 0.1 > param_.affine_param().max_translation()) {
        h_off_jiggle = Rand(param_.affine_param().max_translation() * 2) - param_.affine_param().max_translation();
        w_off_jiggle = Rand(param_.affine_param().max_translation() * 2) - param_.affine_param().max_translation();
      } else {
        h_off_jiggle = Rand(cv_img_transformed.rows * 0.1 * 2) - cv_img_transformed.rows * 0.1; 
        w_off_jiggle = Rand(cv_img_transformed.cols * 0.1 * 2) - cv_img_transformed.cols * 0.1; 
      } 
    }

    cv::Rect roi(std::max(w_off_jiggle, 0), 
                 std::max(h_off_jiggle, 0), 
                 cv_img_transformed_bak.cols - std::abs(w_off_jiggle), 
                 cv_img_transformed_bak.rows - std::abs(h_off_jiggle));
    cv_img_transformed = cv_img_transformed_bak(roi);
  }
 
  // do resize to new_size
  cv::Mat cv_img; 
  cv::resize(cv_img_transformed, cv_img, cv::Size(new_width, new_height), 0, 0, cv::INTER_CUBIC);
  //cv::imwrite("/media/data/kaggle/nationalDataScienceBowl/models/caffe_data_aug/img_transformed.jpg", cv_img); 

  Dtype* mean = NULL;
  if (has_mean_file) {
    CHECK_EQ(img_channels, data_mean_.channels());
    CHECK_EQ(new_height, data_mean_.height()); // CHECK_EQ(img_height, data_mean_.height());
    CHECK_EQ(new_width, data_mean_.width()); // CHECK_EQ(img_width, data_mean_.width());
    mean = data_mean_.mutable_cpu_data();
  }
  if (has_mean_values) {
    CHECK(mean_values_.size() == 1 || mean_values_.size() == img_channels) <<
     "Specify either 1 mean_value or as many as channels: " << img_channels;
    if (img_channels > 1 && mean_values_.size() == 1) {
      // Replicate the mean_value for simplicity
      for (int c = 1; c < img_channels; ++c) {
        mean_values_.push_back(mean_values_[0]);
      }
    }
  }

  int h_off = 0;
  int w_off = 0;
  cv::Mat cv_cropped_img = cv_img;
  if (crop_size) {
    CHECK_EQ(crop_size, height);
    CHECK_EQ(crop_size, width);
    // We only do random crop when we do training.
    if (phase_ == Caffe::TRAIN) {
      h_off = Rand(/*img_height*/new_height - crop_size + 1);
      w_off = Rand(/*img_width*/new_width - crop_size + 1);
    } else { // center crop
      h_off = (/*img_height*/new_height - crop_size) / 2;
      w_off = (/*img_width*/new_width - crop_size) / 2;
    }
    cv::Rect roi(w_off, h_off, crop_size, crop_size);
    cv_cropped_img = cv_img(roi);
  } else {
    CHECK_EQ(/*img_height*/new_height, height);
    CHECK_EQ(/*img_width*/new_width, width);
  }

  CHECK(cv_cropped_img.data);

  Dtype* transformed_data = transformed_blob->mutable_cpu_data();
  int top_index;
  for (int h = 0; h < height; ++h) {
    const uchar* ptr = cv_cropped_img.ptr<uchar>(h);
    int img_index = 0;
    for (int w = 0; w < width; ++w) {
      for (int c = 0; c < img_channels; ++c) {
        if (do_mirror) {
          top_index = (c * height + h) * width + (width - 1 - w);
        } else {
          top_index = (c * height + h) * width + w;
        }
        // int top_index = (c * height + h) * width + w;
        Dtype pixel = static_cast<Dtype>(ptr[img_index++]);
        if (has_mean_file) {
          int mean_index = (c * /*img_height*/new_height + h_off + h) * /*img_width*/new_width + w_off + w;
          transformed_data[top_index] =
            (pixel - mean[mean_index]) * scale;
        } else {
          if (has_mean_values) {
            transformed_data[top_index] =
              (pixel - mean_values_[c]) * scale;
          } else {
            transformed_data[top_index] = pixel * scale;
          }
        }
      }
    }
  }
}


template<typename Dtype>
void DataTransformer<Dtype>::TransformAffine(const cv::Mat& cv_img_origin, 
                                       int new_height, int new_width,
                                       Blob<Dtype>* transformed_blob, int batch_id, int image_id) {
  CHECK_GT(new_height, 0); 
  CHECK_GT(new_width, 0); 
  
  cv::Mat cv_img_tmp; 
  cv::resize(cv_img_origin, cv_img_tmp, cv::Size(new_width, new_height), 0, 0, cv::INTER_CUBIC);
  char filename[1024]; // The filename.
  // Put "file" then k then ".txt" in to filename.
  snprintf(filename, sizeof(char) * 1024, 
      "/media/data/kaggle/nationalDataScienceBowl/models/caffe_data_aug/tmp/%d_%d_img_origin.jpg", batch_id, image_id);
  cv::imwrite(filename, cv_img_tmp); 

  const int img_channels = cv_img_origin.channels();
  const int img_height = cv_img_origin.rows;
  const int img_width = cv_img_origin.cols;

  const int channels = transformed_blob->channels();
  const int height = transformed_blob->height();
  const int width = transformed_blob->width();
  const int num = transformed_blob->num();

  CHECK_LE(height, new_height); // CHECK_LE(height, img_height);
  CHECK_LE(width, new_width); // CHECK_LE(width, img_width);
  CHECK_EQ(channels, img_channels);
  CHECK_GE(num, 1);

  CHECK(cv_img_origin.depth() == CV_8U) << "Image data type must be unsigned byte";

  const int crop_size = param_.crop_size();
  const Dtype scale = param_.scale();
  const bool do_mirror = param_.mirror() && Rand(2);
  const bool has_mean_file = param_.has_mean_file();
  const bool has_mean_values = mean_values_.size() > 0;

  CHECK_GT(img_channels, 0);
  CHECK_GE(new_height, crop_size); // CHECK_GE(img_height, crop_size); 
  CHECK_GE(new_width, crop_size); // CHECK_GE(img_width, crop_size);

  // make transform matrix
  cv::Mat cv_img_rot, cv_img_transformed;  
  // initialize transformation parameter
  float backgroundColor = param_.affine_param().background_color();
  // rotation first
  float angle = 0;
  caffe_rng_uniform<float>(1,
                           -param_.affine_param().rotation_unirnd_range(),
                           param_.affine_param().rotation_unirnd_range(),
                           &angle);
  //printf("angle: %.2f\n", angle); 
  cv::Mat rot = cv::getRotationMatrix2D(cv::Point2f(img_width/2.f, img_height/2.f), angle, 1.0); 
  cv::Size rs = rotatedImageSize(rot, cv::Rect(0,0, img_width, img_height));
  cv_img_rot.create(rs,cv_img_origin.type());
  cv::warpAffine( cv_img_origin, cv_img_rot, rot, cv_img_transformed.size(), cv::INTER_CUBIC, cv::BORDER_CONSTANT, cv::Scalar::all(backgroundColor));
  // stretch
  float tmp = 0;
  caffe_rng_uniform<float>(1,
                           -param_.affine_param().x_stretch_unirnd_range(),
                           param_.affine_param().x_stretch_unirnd_range(),
                           &tmp);
  float x_stretch=1+tmp; // x stretch
  caffe_rng_uniform<float>(1,
                           -param_.affine_param().y_stretch_unirnd_range(),
                           param_.affine_param().y_stretch_unirnd_range(),
                           &tmp);
  float y_stretch=1+tmp; // y stretch
  cv::resize(cv_img_rot, cv_img_transformed, 
             cv::Size(cv_img_rot.cols * x_stretch, cv_img_rot.rows * y_stretch), 0, 0, cv::INTER_CUBIC);

  ////// affine until here

  // do jiggle
  cv::Mat cv_img_transformed_bak; 
  if (param_.affine_param().has_max_translation()) {
    cv_img_transformed_bak = cv_img_transformed;

    int h_off_jiggle = 0; 
    int w_off_jiggle = 0; 
    
    if (param_.affine_param().max_translation() > 0) {
      if (cv_img_transformed.rows * 0.1 > param_.affine_param().max_translation()) {
        h_off_jiggle = Rand(param_.affine_param().max_translation() * 2) - param_.affine_param().max_translation();
        w_off_jiggle = Rand(param_.affine_param().max_translation() * 2) - param_.affine_param().max_translation();
      } else {
        h_off_jiggle = Rand(cv_img_transformed.rows * 0.1 * 2) - cv_img_transformed.rows * 0.1; 
        w_off_jiggle = Rand(cv_img_transformed.cols * 0.1 * 2) - cv_img_transformed.cols * 0.1; 
      } 
    }
    cv::Rect roi(std::max(w_off_jiggle, 0), 
                 std::max(h_off_jiggle, 0), 
                 cv_img_transformed_bak.cols - std::abs(w_off_jiggle), 
                 cv_img_transformed_bak.rows - std::abs(h_off_jiggle));
    cv_img_transformed = cv_img_transformed_bak(roi);
  }
 
  // do resize to new_size
  cv::Mat cv_img; 
  cv::resize(cv_img_transformed, cv_img, cv::Size(new_width, new_height), 0, 0, cv::INTER_CUBIC);
  snprintf(filename, sizeof(char) * 1024, 
      "/media/data/kaggle/nationalDataScienceBowl/models/caffe_data_aug/tmp/%d_%d_image_transformed.jpg", batch_id, image_id);
  cv::imwrite(filename, cv_img);

  Dtype* mean = NULL;
  if (has_mean_file) {
    CHECK_EQ(img_channels, data_mean_.channels());
    CHECK_EQ(new_height, data_mean_.height()); // CHECK_EQ(img_height, data_mean_.height());
    CHECK_EQ(new_width, data_mean_.width()); // CHECK_EQ(img_width, data_mean_.width());
    mean = data_mean_.mutable_cpu_data();
  }
  if (has_mean_values) {
    CHECK(mean_values_.size() == 1 || mean_values_.size() == img_channels) <<
     "Specify either 1 mean_value or as many as channels: " << img_channels;
    if (img_channels > 1 && mean_values_.size() == 1) {
      // Replicate the mean_value for simplicity
      for (int c = 1; c < img_channels; ++c) {
        mean_values_.push_back(mean_values_[0]);
      }
    }
  }

  int h_off = 0;
  int w_off = 0;
  cv::Mat cv_cropped_img = cv_img;
  if (crop_size) {
    CHECK_EQ(crop_size, height);
    CHECK_EQ(crop_size, width);
    // We only do random crop when we do training.
    if (phase_ == Caffe::TRAIN) {
      h_off = Rand(/*img_height*/new_height - crop_size + 1);
      w_off = Rand(/*img_width*/new_width - crop_size + 1);
    } else { // center crop
      h_off = (/*img_height*/new_height - crop_size) / 2;
      w_off = (/*img_width*/new_width - crop_size) / 2;
    }
    cv::Rect roi(w_off, h_off, crop_size, crop_size);
    cv_cropped_img = cv_img(roi);
  } else {
    CHECK_EQ(/*img_height*/new_height, height);
    CHECK_EQ(/*img_width*/new_width, width);
  }

  CHECK(cv_cropped_img.data);

  Dtype* transformed_data = transformed_blob->mutable_cpu_data();
  int top_index;
  for (int h = 0; h < height; ++h) {
    const uchar* ptr = cv_cropped_img.ptr<uchar>(h);
    int img_index = 0;
    for (int w = 0; w < width; ++w) {
      for (int c = 0; c < img_channels; ++c) {
        if (do_mirror) {
          top_index = (c * height + h) * width + (width - 1 - w);
        } else {
          top_index = (c * height + h) * width + w;
        }
        // int top_index = (c * height + h) * width + w;
        Dtype pixel = static_cast<Dtype>(ptr[img_index++]);
        if (has_mean_file) {
          int mean_index = (c * /*img_height*/new_height + h_off + h) * /*img_width*/new_width + w_off + w;
          transformed_data[top_index] =
            (pixel - mean[mean_index]) * scale;
        } else {
          if (has_mean_values) {
            transformed_data[top_index] =
              (pixel - mean_values_[c]) * scale;
          } else {
            transformed_data[top_index] = pixel * scale;
          }
        }
      }
    }
  }
}


template<typename Dtype>
void DataTransformer<Dtype>::Transform(Blob<Dtype>* input_blob,
                                       Blob<Dtype>* transformed_blob) {
  const int input_num = input_blob->num();
  const int input_channels = input_blob->channels();
  const int input_height = input_blob->height();
  const int input_width = input_blob->width();

  const int num = transformed_blob->num();
  const int channels = transformed_blob->channels();
  const int height = transformed_blob->height();
  const int width = transformed_blob->width();
  const int size = transformed_blob->count();

  CHECK_LE(input_num, num);
  CHECK_EQ(input_channels, channels);
  CHECK_GE(input_height, height);
  CHECK_GE(input_width, width);

  const int crop_size = param_.crop_size();
  const Dtype scale = param_.scale();
  const bool do_mirror = param_.mirror() && Rand(2);
  const bool has_mean_file = param_.has_mean_file();
  const bool has_mean_values = mean_values_.size() > 0;

  int h_off = 0;
  int w_off = 0;
  if (crop_size) {
    CHECK_EQ(crop_size, height);
    CHECK_EQ(crop_size, width);
    // We only do random crop when we do training.
    if (phase_ == Caffe::TRAIN) {
      h_off = Rand(input_height - crop_size + 1);
      w_off = Rand(input_width - crop_size + 1);
    } else {
      h_off = (input_height - crop_size) / 2;
      w_off = (input_width - crop_size) / 2;
    }
  } else {
    CHECK_EQ(input_height, height);
    CHECK_EQ(input_width, width);
  }

  Dtype* input_data = input_blob->mutable_cpu_data();
  if (has_mean_file) {
    CHECK_EQ(input_channels, data_mean_.channels());
    CHECK_EQ(input_height, data_mean_.height());
    CHECK_EQ(input_width, data_mean_.width());
    for (int n = 0; n < input_num; ++n) {
      int offset = input_blob->offset(n);
      caffe_sub(data_mean_.count(), input_data + offset,
            data_mean_.cpu_data(), input_data + offset);
    }
  }

  if (has_mean_values) {
    CHECK(mean_values_.size() == 1 || mean_values_.size() == input_channels) <<
     "Specify either 1 mean_value or as many as channels: " << input_channels;
    if (mean_values_.size() == 1) {
      caffe_add_scalar(input_blob->count(), -(mean_values_[0]), input_data);
    } else {
      for (int n = 0; n < input_num; ++n) {
        for (int c = 0; c < input_channels; ++c) {
          int offset = input_blob->offset(n, c);
          caffe_add_scalar(input_height * input_width, -(mean_values_[c]),
            input_data + offset);
        }
      }
    }
  }

  Dtype* transformed_data = transformed_blob->mutable_cpu_data();

  for (int n = 0; n < input_num; ++n) {
    int top_index_n = n * channels;
    int data_index_n = n * channels;
    for (int c = 0; c < channels; ++c) {
      int top_index_c = (top_index_n + c) * height;
      int data_index_c = (data_index_n + c) * input_height + h_off;
      for (int h = 0; h < height; ++h) {
        int top_index_h = (top_index_c + h) * width;
        int data_index_h = (data_index_c + h) * input_width + w_off;
        if (do_mirror) {
          int top_index_w = top_index_h + width - 1;
          for (int w = 0; w < width; ++w) {
            transformed_data[top_index_w-w] = input_data[data_index_h + w];
          }
        } else {
          for (int w = 0; w < width; ++w) {
            transformed_data[top_index_h + w] = input_data[data_index_h + w];
          }
        }
      }
    }
  }
  if (scale != Dtype(1)) {
    DLOG(INFO) << "Scale: " << scale;
    caffe_scal(size, scale, transformed_data);
  }
}

template <typename Dtype>
void DataTransformer<Dtype>::InitRand() {
  //const bool needs_rand = param_.mirror() ||
  //    (phase_ == Caffe::TRAIN && param_.crop_size());
  const bool needs_rand = 1; 
  if (needs_rand) {
    const unsigned int rng_seed = caffe_rng_rand();
    rng_.reset(new Caffe::RNG(rng_seed));
  } else {
    rng_.reset();
  }
}

template <typename Dtype>
int DataTransformer<Dtype>::Rand(int n) {
  CHECK(rng_);
  CHECK_GT(n, 0);
  caffe::rng_t* rng =
      static_cast<caffe::rng_t*>(rng_->generator());
  return ((*rng)() % n);
}

INSTANTIATE_CLASS(DataTransformer);

}  // namespace caffe
