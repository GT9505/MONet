#include <algorithm>
#include <cfloat>
#include <vector>



#include "caffe/layers/my_choose_layer.hpp"
#include "caffe/util/gpu_util.cuh"

namespace caffe {


template <typename Dtype>
__global__ void FindChoose(const int nthreads, const Dtype* bottom_data_1,
    Dtype* choose_dim,Dtype* max_val_) {
  CUDA_KERNEL_LOOP(index, nthreads) { 
    for(int i=0;i<4;++i){
  
	  if (bottom_data_1[index*4+i] > max_val_[index]) {
	  
          max_val_[index] = bottom_data_1[index*4+i];
		  
          choose_dim[index] = i;
		  
		  }
  }
  }
}

template <typename Dtype>
__global__ void MyChoose(const int nthreads, const Dtype* bottom_data,
    const Dtype* choose_dim,Dtype* top_data) {
  CUDA_KERNEL_LOOP(index, nthreads) { 
    top_data[index] = bottom_data[(index/(40*49))*160*49+40*int(choose_dim[index/(40*49)])*49+index % (40*49)];
  }
}

template <typename Dtype>
void MyChooseLayer<Dtype>::Forward_gpu(
    const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
  
  const Dtype* bottom_data = bottom[0]->cpu_data();
  
  const Dtype* bottom_data_1 = bottom[1]->cpu_data();
  
  Dtype* top_data = top[0]->mutable_cpu_data();  
  
  const int nthreads = top[0]->count();
  
  caffe_gpu_set(nthreads, Dtype(0), top_data);
  caffe_gpu_set(max_val_.count(), Dtype(-FLT_MAX), max_val_.mutable_cpu_data());
  caffe_gpu_set(choose_dim.count(), Dtype(-1), choose_dim.mutable_cpu_data());
  
  FindChoose<Dtype>  // NOLINT_NEXT_LINE(whitespace/operators)
      <<<CAFFE_GET_BLOCKS(nthreads/(40*49)), CAFFE_CUDA_NUM_THREADS>>>(
      nthreads/(40*49), bottom_data_1, choose_dim.mutable_cpu_data(),max_val_.mutable_cpu_data());
  
  MyChoose<Dtype>  // NOLINT_NEXT_LINE(whitespace/operators)
      <<<CAFFE_GET_BLOCKS(nthreads), CAFFE_CUDA_NUM_THREADS>>>(
      nthreads, bottom_data, choose_dim.cpu_data(),top_data);
  
}





template <typename Dtype>
__global__ void MySumBackward(const int nthreads, const Dtype* top_diff,
     Dtype* bottom_diff) {
  CUDA_KERNEL_LOOP(index, nthreads) {	
	for (int i=0;i<40*49;++i){
		bottom_diff[index]+=top_diff[index*40*49+i];
	}
  }
}

template <typename Dtype>
__global__ void MyChooseBackward(const int nthreads, const Dtype* top_diff,
    const Dtype* choose_dim, Dtype* bottom_diff, Dtype* bottom_diff1, const Dtype* choose_dim_diff) {
  CUDA_KERNEL_LOOP(index, nthreads) {	
	if (index /(40*49) % 4 == choose_dim[index/(160*49)]){
		bottom_diff[index]=top_diff[(index/(160*49))*40*49+index%(40*49)];
	}
	bottom_diff1[4*(index/(160*49))+int(choose_dim[index/(160*49)])]=choose_dim_diff[index/(160*49)];
  }
}

template <typename Dtype>
void MyChooseLayer<Dtype>::Backward_gpu(const vector<Blob<Dtype>*>& top,
    const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {
  if (!propagate_down[0]) { return; }
  const Dtype* top_diff = top[0]->gpu_diff();
  Dtype* bottom_diff = bottom[0]->mutable_gpu_diff();
  Dtype* bottom_diff1 = bottom[1]->mutable_gpu_diff();
  const int nthreads = bottom[0]->count();
  caffe_gpu_set(nthreads, Dtype(0), bottom_diff);
  caffe_gpu_set(nthreads/(40*49), Dtype(0), bottom_diff1);
  caffe_gpu_set(choose_dim.count(),Dtype(0),choose_dim.mutable_gpu_diff());// caculate the sum of top[0]
  
  MySumBackward<Dtype>  // NOLINT_NEXT_LINE(whitespace/operators)
      <<<CAFFE_GET_BLOCKS(choose_dim.count()), CAFFE_CUDA_NUM_THREADS>>>(
      choose_dim.count(), top_diff ,choose_dim.mutable_gpu_diff());
  
  MyChooseBackward<Dtype>  // NOLINT_NEXT_LINE(whitespace/operators)
      <<<CAFFE_GET_BLOCKS(nthreads), CAFFE_CUDA_NUM_THREADS>>>(
      nthreads, top_diff ,choose_dim.cpu_data(), bottom_diff,bottom_diff1,choose_dim.gpu_diff());
}

INSTANTIATE_LAYER_GPU_FUNCS(MyChooseLayer);

}  // namespace caffe
