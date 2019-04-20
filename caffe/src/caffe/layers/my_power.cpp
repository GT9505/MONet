#include <algorithm>
#include <string>
#include <utility>
#include <vector>

#include "caffe/util/math_functions.hpp"

#include "caffe/layers/my_power.hpp" 

namespace caffe {
template <typename Dtype>
void MyPowerLayer<Dtype>::LayerSetUp(const vector<Blob<Dtype>*>& bottom,
		const vector<Blob<Dtype>*>& top){
			//printf ("7\n");
		}


template <typename Dtype>
void MyPowerLayer<Dtype>::Reshape(const vector<Blob<Dtype>*>& bottom,
		const vector<Blob<Dtype>*>& top){
   //printf ("8\n");
	top[0]->Reshape(bottom[0]->num(),bottom[0]->channels(),bottom[0]->height(),bottom[0]->width());
	for (int i = 0; i < top[0]->count(); ++i) {
		top[0]->mutable_cpu_data()[i] = Dtype(0);
	}

}
		
template <typename Dtype>
void MyPowerLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top){
	NOT_IMPLEMENTED;
}

template <typename Dtype>
void MyPowerLayer<Dtype>::Backward_cpu(
  const vector<Blob<Dtype>*>& top, const vector<bool>& propagate_down,
  const vector<Blob<Dtype>*>& bottom) {
	NOT_IMPLEMENTED;
}

#ifdef CPU_ONLY
STUB_GPU(MyPowerLayer);
#endif

INSTANTIATE_CLASS(MyPowerLayer);
REGISTER_LAYER_CLASS(MyPower);

}  // namespace caffe