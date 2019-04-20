#include "caffe/layers/multi_class_cross_entropy_loss_layer.hpp"
namespace caffe {
template <typename Dtype>
__global__ void MulticlassCrossEntropyLossForwardGPU(const int nthreads,
          const Dtype* input_data, const Dtype* label, Dtype* loss,Dtype* weights,const Dtype sum_labels,const int channels) {
  CUDA_KERNEL_LOOP(index, nthreads) {
    Dtype tmp = input_data[index];
    if (tmp < 1e-10) {
      tmp = 1e-10;
    }
    if (tmp > (1 - 1e-10)) {
      tmp = 1 - 1e-10;
    }
    if (int(label[index]) == 0) {
      //weights[index] = Dtype(0.8130);
      weights[index] = Dtype(3*channels/Dtype(4))/(channels-sum_labels);
      loss[index] = -log(1 - tmp);
    }
    else {
      //weights[index] = Dtype(3.2258);
      weights[index] = Dtype(channels/Dtype(4))/sum_labels;
      loss[index] = -log(tmp);
    }
  }
}
template <typename Dtype>
void MulticlassCrossEntropyLossLayer<Dtype>::Forward_gpu(
    const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
  const Dtype* input_data = bottom[0]->gpu_data();
  const Dtype* label = bottom[1]->gpu_data();
  const int nthreads = batch_size * channels;
  Dtype sum_labels=0;

  //printf("%f\t%f\t%f\t%f\t%f\t%f\t%f\t%f\t%f\t%f\t%f\t%f\t%f\t%f\t%f\t%f\t%f\t%f\t%f\t%f\n",input_data[0],input_data[1],input_data[2],input_data[3],input_data[4],input_data[5],input_data[6],input_data[7],input_data[8],input_data[9],input_data[10],input_data[11],input_data[12],input_data[13],input_data[14],input_data[15],input_data[16],input_data[17],input_data[18],input_data[19]);

  caffe_gpu_asum(channels, label, &sum_labels);
  /*for(i=0;i<channels;i++){
      printf("%d\n",i);
      //sum_labels+=static_cast<int>(label[i]);
      }*/

  //printf ("%d\t%d\t%d\t%d\n",bottom[0]->num(),bottom[0]->channels(),bottom[0]->height(),bottom[0]->width());
  //printf ("%d\t%d\t%d\t%d\n",bottom[1]->num(),bottom[1]->channels(),bottom[1]->height(),bottom[1]->width());
  // Since this memory is not used for anything until it is overwritten
  // on the backward pass, we use it here to avoid having to allocate new GPU
  // memory to accumulate intermediate results in the kernel.
  Dtype* loss_data = bottom[0]->mutable_gpu_diff();     
  // Similarly, this memory is never used elsewhere, and thus we can use it
  // to avoid having to allocate additional GPU memory.
  // NOLINT_NEXT_LINE(whitespace/operators)

  MulticlassCrossEntropyLossForwardGPU<Dtype><<<CAFFE_GET_BLOCKS(nthreads),
      CAFFE_CUDA_NUM_THREADS>>>(nthreads, input_data, label, loss_data,weights.mutable_gpu_data(),sum_labels,channels);

  Dtype loss;
  caffe_gpu_asum(nthreads, loss_data, &loss);

  loss /= batch_size;

  top[0]->mutable_cpu_data()[0] = loss;

}
template <typename Dtype>
__global__ void MulticlassCrossEntropyLossBackwardGPU(const int nthreads, 
          const Dtype* input_data, const Dtype* label, Dtype* bottom_diff, 
                  const int channels,const Dtype* weights) {
  CUDA_KERNEL_LOOP(index, nthreads) {
    Dtype tmp = input_data[index];
    if (tmp < 1e-10) {
      tmp = 1e-10;
    }
    if (tmp > (1 - 1e-10)) {
      tmp = 1 - 1e-10;
    }
    if (int(label[index]) == 0) {
      bottom_diff[index] = 1.0 / (channels * (1 - tmp));
    }
    else {
      bottom_diff[index] = -1.0/ (channels * tmp);
    }
  }
}
template <typename Dtype>
void MulticlassCrossEntropyLossLayer<Dtype>::Backward_gpu(const vector<Blob<Dtype>*>& top,
    const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {
  if (propagate_down[1]) {
    LOG(FATAL) << this->type()
               << " Layer cannot backpropagate to label inputs.";
  }
  if (propagate_down[0]) {
    Dtype* bottom_diff = bottom[0]->mutable_gpu_diff();
    const Dtype* input_data = bottom[0]->gpu_data();
    const Dtype* label = bottom[1]->gpu_data();
    const int nthreads = batch_size * channels;

    MulticlassCrossEntropyLossBackwardGPU<Dtype><<<CAFFE_GET_BLOCKS(nthreads),
        CAFFE_CUDA_NUM_THREADS>>>(nthreads, input_data, label, bottom_diff,
        channels,weights.gpu_data());

    const Dtype loss_weight = top[0]->cpu_diff()[0];
    
    caffe_gpu_scal(bottom[0]->count(), loss_weight / batch_size, bottom_diff);
  }
}
INSTANTIATE_LAYER_GPU_FUNCS(MulticlassCrossEntropyLossLayer);
}  // namespace caffe