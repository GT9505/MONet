#ifndef CAFFE_MY_POWER_HPP_ 
#define CAFFE_MY_POWER_HPP_  

#include <vector>

#include "caffe/blob.hpp"  
#include "caffe/layer.hpp"  
#include "caffe/proto/caffe.pb.h"  
#include "caffe/common.hpp"  

namespace caffe {
	//written by gongtao 2017/6/25
template <typename Dtype>
class MyPowerLayer : public Layer<Dtype>{
  public:
    explicit MyPowerLayer(const LayerParameter& param)
		: Layer<Dtype>(param) {}
	virtual void LayerSetUp(const vector<Blob<Dtype>*>& bottom,  
      const vector<Blob<Dtype>*>& top);  
    virtual void Reshape(const vector<Blob<Dtype>*>& bottom,  
      const vector<Blob<Dtype>*>& top);
	  
    virtual inline const char* type() const { return "MyPower"; }  
	
    virtual inline int ExactNumBottomBlobs() const { return -1; }
    virtual inline int MinBottomBlobs() const { return 1; }
    virtual inline int MaxBottomBlobs() const { return 2; }  
    virtual inline int ExactNumTopBlobs() const { return 1; } 
	
  protected:
   virtual void Forward_cpu(const vector<Blob<Dtype>*>& bottom,  
      const vector<Blob<Dtype>*>& top); 
   virtual void Forward_gpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);
	  
   virtual void Backward_cpu(const vector<Blob<Dtype>*>& top,  
      const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) ; 
   virtual void Backward_gpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) ;	
       

};
} // namespace caffe
#endif  // CAFFE_PROPOSAL_TO_PREDBBOX_LAYER_HPP_  