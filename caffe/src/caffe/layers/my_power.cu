#include <algorithm>
#include <cfloat>
#include <vector>

#include "thrust/device_vector.h"

#include "caffe/util/math_functions.hpp"

#include "caffe/layers/my_power.hpp" 

namespace caffe {	
	template <typename Dtype>
	void MyPowerLayer<Dtype>::Forward_gpu(const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top){
		const Dtype* bottom_data = bottom[0]->cpu_data();
		Dtype* top_data = top[0]->mutable_cpu_data();
   //int flag=0;
   
		for (int i=0;i<bottom[0]->count();i++){
		top_data[i]=bottom_data[i]*bottom[0]->height();
		}
   //printf ("%d\n",bottom[0]->height());
   //printf("%d\n",flag);
	}
	template <typename Dtype>
	void MyPowerLayer<Dtype>::Backward_gpu(const vector<Blob<Dtype>*>& top, const vector<bool>& propagate_down,
		const vector<Blob<Dtype>*>& bottom){
   if (!propagate_down[0]) {
    return;
  }
		Dtype* bottom_diff = bottom[0]->mutable_cpu_diff();
    const int count = bottom[0]->count();
    //printf ("%d\n",count);
    const Dtype* top_diff = top[0]->cpu_diff();
    caffe_gpu_set(count, Dtype(0.), bottom_diff);
    //printf("sdfdsfs\n");
    for (int i=0;i<count;i++){
        bottom_diff[i]=top_diff[i]*bottom[0]->height();	
        }	
	}

	INSTANTIATE_LAYER_GPU_FUNCS(MyPowerLayer);
// namespace caffe
}