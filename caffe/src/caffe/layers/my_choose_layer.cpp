#include <cfloat>
#include <algorithm>

#include <string>
#include <utility>
#include <vector>

#include "caffe/layers/my_choose_layer.hpp"
#include "caffe/util/math_functions.hpp"

namespace caffe {

template <typename Dtype>
void MyChooseLayer<Dtype>::LayerSetUp(const vector<Blob<Dtype>*>& bottom,
		const vector<Blob<Dtype>*>& top){
			
		}
		
template <typename Dtype>
void MyChooseLayer<Dtype>::Reshape(
    const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
		
	top[0]->Reshape(bottom[0]->num(),bottom[0]->channels()/bottom[1]->channels(),bottom[0]->height(),bottom[0]->width());
	choose_dim.Reshape(bottom[0]->num(),1,1,1);
	max_val_.Reshape(bottom[0]->num(),1,1,1);
}

template <typename Dtype>
void MyChooseLayer<Dtype>::Forward_cpu(
    const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
	NOT_IMPLEMENTED;
}

template <typename Dtype>
void MyChooseLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
    const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {
	NOT_IMPLEMENTED;
}

#ifdef CPU_ONLY
STUB_GPU(MyChooseLayer);
#endif

INSTANTIATE_CLASS(MyChooseLayer);
REGISTER_LAYER_CLASS(MyChoose);

}  // namespace caffe
