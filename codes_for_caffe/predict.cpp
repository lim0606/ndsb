
//Jaehyun Lim
#include <glog/logging.h>

#include <cstring>
#include <map>
#include <string>
#include <vector>
#include <fstream> 

#include "hdf5.h"
#include "leveldb/db.h"
#include "lmdb.h"

#include "caffe/caffe.hpp"

using caffe::Blob;
using caffe::Caffe;
using caffe::Net;
using caffe::Layer;
using caffe::shared_ptr;
using caffe::Timer;
using caffe::vector;

// Define flags
DEFINE_int32(gpu, -1,
    "Run in GPU mode on given device ID.");
//DEFINE_string(solver, "",
//    "The solver definition protocol buffer text file.");
DEFINE_string(model, "",
    "The model definition protocol buffer text file..");
DEFINE_string(snapshot, "",
    "The snapshot solver state to resume training.");
DEFINE_string(weights, "",
    "The pretrained weights to initialize finetuning. "
    "Cannot be set simultaneously with snapshot.");
//DEFINE_int32(iterations, 50,
//    "The number of iterations to run.");
//DEFINE_int32(numdata, 0,
//    "The total number of test data. (you should specify in this implementation)."); 
//DEFINE_int32(batchsize, 0,
//    "The batchsize. (you should specify in this implementation)."); 
DEFINE_string(labellist, "",
    "The text file having labels and their corresponding indices.");
DEFINE_string(outfile, "",
    "The text file including prediction probabilities.");


int main(int argc, char** argv) {
  // Print output to stderr (while still logging).
  FLAGS_alsologtostderr = 1;
  // Usage message.
  gflags::SetUsageMessage("\n"
      "usage: predict <args>\n\n");
  // Run tool or show usage.
  caffe::GlobalInit(&argc, &argv);
  if (argc == 2) {
//    return GetBrewFunction(caffe::string(argv[1]))();
  } else {
    gflags::ShowUsageWithFlagsRestrict(argv[0], "tools/predict");
  }

  // label (open label txt for label names)
  std::ifstream label_file;
  label_file.open(FLAGS_labellist.c_str());
  if(!label_file) {
    printf("Please specify the label list file. For example, ndsb_labels.txt.\n"); 
    return 0;  
  }
  std::vector< std::string > label_names;
  std::vector< int > label_indices;
  std::string label_name;
  int label_index; 
  int num_classes;
  while(label_file >> label_index >> label_name) {
//    printf("label_index: %d, label_name: %s\n", label_index, label_name.c_str()); 
    label_names.push_back(label_name); 
    label_indices.push_back(label_index); 
  }
  num_classes = label_indices.size(); 
//  printf("# of classes : %d\n", num_classes); 

  //
  CHECK_GT(FLAGS_model.size(), 0) << "Need a model definition to predict.";
  CHECK_GT(FLAGS_weights.size(), 0) << "Need model weights to predict.";

  // Set device id and mode
  if (FLAGS_gpu >= 0) {
    LOG(INFO) << "Use GPU with device ID " << FLAGS_gpu;
    Caffe::SetDevice(FLAGS_gpu);
    Caffe::set_mode(Caffe::GPU);
  } else {
    LOG(INFO) << "Use CPU.";
    Caffe::set_mode(Caffe::CPU);
  }
  // Instantiate the caffe net.
  Caffe::set_phase(Caffe::TEST);
  Net<float> caffe_net(FLAGS_model);
  caffe_net.CopyTrainedLayersFrom(FLAGS_weights);

  // Calculate iterations
/*  if (FLAGS_numdata == 0) {
    printf("Please specify total number of data.\n");
    return 0;  
  }
  if (FLAGS_batchsize == 0) {
    printf("Please specify batchsize.\n");
    return 0;  
  } 
  LOG(INFO) << "num data: " << FLAGS_numdata << ", batchsize: " << FLAGS_batchsize;
  int iterations =(int)( (float)FLAGS_numdata / (float)FLAGS_batchsize ) + 1;
  LOG(INFO) << "# of iterations " << iterations; 
  //LOG(INFO) << "Running for " << FLAGS_iterations << " iterations.";
*/
  int iterations = -1, numdata = -1, batchsize = -1;
  const vector<shared_ptr<Layer<float> > >& layers = caffe_net.layers();
  LOG(INFO) << "# of layers " <<  (int)layers.size();
  for (int i = 0; i < layers.size(); ++i) {
    const caffe::string& layername = layers[i]->layer_param().name();
    LOG(INFO) << std::setfill(' ') << std::setw(10) << layername;
  }
  LOG(INFO) << "layer type: " << layers[0]->layer_param().type(); 
  switch (layers[0]->layer_param().type()) {
    case 5: {// DATA
      batchsize = layers[0]->layer_param().data_param().batch_size();
      //LOG(INFO) << "batch_size: " << batch_size;
      int backend = (int)layers[0]->layer_param().data_param().backend();
      LOG(INFO) << "backend (LEVELDB: 0, LMDB:1): " << backend;

      if (backend == 1) { // LMDB
        MDB_env* mdb_env;
        MDB_stat mdb_mst;
        CHECK_EQ(mdb_env_create(&mdb_env), MDB_SUCCESS) << "mdb_env_create failed";
        CHECK_EQ(mdb_env_set_mapsize(mdb_env, 1099511627776), MDB_SUCCESS);  // 1TB
        CHECK_EQ(mdb_env_open(mdb_env,
             layers[0]->layer_param().data_param().source().c_str(),
             MDB_RDONLY|MDB_NOTLS, 0664), MDB_SUCCESS) << "mdb_env_open failed";
        (void)mdb_env_stat(mdb_env, &mdb_mst);

        //LOG(INFO) << "FINALLY!!! # of images: " << mdb_mst.ms_entries;
        numdata = mdb_mst.ms_entries;
        
      } else { // LEVELDB
        LOG(INFO) << "LEVELDB is currently not supported. sorry :)"; 
        return 0;
      }
      break;
    }
    case 12: { // IMAGE_DATA
      batchsize = layers[0]->layer_param().image_data_param().batch_size();
      LOG(INFO) << "batch_size: " << batchsize;
      unsigned int number_of_lines = 0;       

      FILE *infile = fopen(layers[0]->layer_param().image_data_param().source().c_str(), "r");
      int ch;

      while (EOF != (ch=getc(infile)))
        if ('\n' == ch)
          ++number_of_lines;
      //printf("%u\n", number_of_lines);
      numdata = (int)number_of_lines; 
      break;
    }
    case 43: { // IMAGE_DATA_AFFINE
      batchsize = layers[0]->layer_param().image_data_affine_param().batch_size();
      LOG(INFO) << "batch_size: " << batchsize;
      unsigned int number_of_lines = 0;

      FILE *infile = fopen(layers[0]->layer_param().image_data_affine_param().source().c_str(), "r");
      int ch;

      while (EOF != (ch=getc(infile)))
        if ('\n' == ch)
          ++number_of_lines;
      //printf("%u\n", number_of_lines);
      numdata = (int)number_of_lines;
      break;
    }
    case 44: { // IMAGE_DATA_MULTIPLE_INFERENCE
      batchsize = layers[0]->layer_param().image_data_multi_infer_param().batch_size();
      LOG(INFO) << "batch_size: " << batchsize;
      unsigned int number_of_lines = 0;

      FILE *infile = fopen(layers[0]->layer_param().image_data_multi_infer_param().source().c_str(), "r");
      int ch;

      while (EOF != (ch=getc(infile)))
        if ('\n' == ch)
          ++number_of_lines;
      //printf("%u\n", number_of_lines);
      numdata = (int)number_of_lines;
      break;
    }
    default: 
      LOG(INFO) << "predict.cpp assumes layers[0] is either DATA or IMAGE_DATA.";
      return 0;
      break;
  }
  if (batchsize == -1 || numdata == -1) {
    LOG(INFO) << "something wrong in reading # of data and batchsize.";
    return 0;
  } else {
    LOG(INFO) << "num data: " << numdata << ", batchsize: " << batchsize;
  }
  iterations =(int)( (float)numdata / (float)batchsize ) + 1;
  LOG(INFO) << "# of iterations " << iterations; 
  //LOG(INFO) << "Running for " << FLAGS_iterations << " iterations.";

  
  FILE *prediction_file;
  prediction_file = fopen(FLAGS_outfile.c_str(), "w"); 
  if (!prediction_file) {
    printf("Please specify the label list file. For example, prediction.txt.\n");
    return 0;  
  }
  
  //printf("# of iterations: %d\n", FLAGS_iterations);
  int img_idx = 0, img_processed_idx = 0;  
  for (int i = 0; i < /*FLAGS_iterations*/ iterations; ++i) {
    if (i % (int)(0.1*iterations) == 0) {
      LOG(INFO) << (float)i / (float)iterations * 100 << "%"; 
    }
//    printf("iter: %d\n", i); 
    const vector<Blob<float>*>& result =
        caffe_net.ForwardPrefilled();
    const vector<int>& result_indices = caffe_net.output_blob_indices();

    //printf("result.size(): %d\n", (int)result.size()); 
    //for (int k = 0; k < result.size(); ++k) {
    //  printf("blob index: %d\n", result_indices[k]);
    //}
    int label_idx = -1, prob_idx = -1;
    label_idx = result_indices[0] < result_indices[1] ? 0 : 1;
    prob_idx = result_indices[0] < result_indices[1] ? 1 : 0;

    // data label (in here -1 for all data, since it is unlabeled data)
    int batchsize = result[label_idx]->count(); 
//    printf("batchsize: %d, num_classes: %d\n", batchsize, num_classes); 
//    printf("result[0][0]: %.3f", result_vec[0]);
//    for (int j = 1; j < batchsize; ++j){
//      printf(", result[0][%d]: %.3f", result_vec[j]); 
//    } 
//    printf("\n"); 

    // prediction probs num_classes x batchsize 
    const float* prob_vec = result[prob_idx]->cpu_data(); 
    for (int j = 0; j < batchsize; ++j){
      if (img_idx < numdata) {
        fprintf(prediction_file, "%e", prob_vec[j*num_classes]); 
        for (int k = 1; k < num_classes; ++k) {
          fprintf(prediction_file, ",%e", prob_vec[j*num_classes+k]); 
        }
        fprintf(prediction_file, "\n"); 
        ++img_processed_idx;
      }
      ++img_idx;
    }
  }
  LOG(INFO) << "100%"; 
  LOG(INFO) << "# of imgs (read): " << img_idx << ", # of imgs (processed): " << img_processed_idx;
  fclose(prediction_file); 

  return 0; 
}
