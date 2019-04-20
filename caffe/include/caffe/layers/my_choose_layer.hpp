#ifndef CAFFE_MY_CHOOSE_LAYER_HPP_
#define CAFFE_MY_CHOOSE_LAYER_HPP_

#include <vector>

#include "caffe/blob.hpp"
#include "caffe/layer.hpp"
#include "caffe/common.hpp"
#include "caffe/proto/caffe.pb.h"

namespace caffe {

/**
 * @brief Copy a Blob along specified dimensions.
 */
template <typename Dtype>
class MyChooseLayer : public Layer<Dtype> {
 public:
  explicit MyChooseLayer(const LayerParameter& param)
      : Layer<Dtype>(param) {}
  virtual void LayerSetUp(const vector<Blob<Dtype>*>& bottom,  
      const vector<Blob<Dtype>*>& top); 
  virtual void Reshape(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);

  virtual inline const char* type() const { return "MyChoose"; }
  virtual inline int ExactNumBottomBlobs() const { return 2; }
  virtual inline int ExactNumTopBlobs() const { return 1; }

 protected:
  virtual void Forward_cpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);
  virtual void Forward_gpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);

  virtual void Backward_cpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom);
  virtual void Backward_gpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom);

  Blob<Dtype>  choose_dim;
  Blob<Dtype>  max_val_;
};

}  // namespace caffe

#endif  // CAFFE_TILE_LAYER_HPP_
