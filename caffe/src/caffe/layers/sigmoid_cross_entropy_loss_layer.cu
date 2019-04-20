#include <vector>

#include "caffe/layers/sigmoid_cross_entropy_loss_layer.hpp"
#include "caffe/util/math_functions.hpp"

namespace caffe {

template <typename Dtype>
__global__ void SigmoidCrossEntropyLossBackwardGPU(const int nthreads, const Dtype* prob_data,
          const Dtype* label, Dtype* bottom_diff, const bool has_ignore_label_,
          const int ignore_label_, Dtype* counts) {

  CUDA_KERNEL_LOOP(index, nthreads) {

    if (has_ignore_label_ && label[index] == ignore_label_) {
      bottom_diff[index] = 0;
      counts[index] = 0;
    } else {
      bottom_diff[index] = prob_data[index] - label[index];
      counts[index] = 1;
    }
  }
}

template <typename Dtype>
void SigmoidCrossEntropyLossLayer<Dtype>::Backward_gpu(
    const vector<Blob<Dtype>*>& top, const vector<bool>& propagate_down,
    const vector<Blob<Dtype>*>& bottom) {
  if (propagate_down[1]) {
    LOG(FATAL) << this->type()
               << " Layer cannot backpropagate to label inputs.";
  }
  if (propagate_down[0]) {
    // First, compute the diff
    const int nthreads = bottom[0]->count();
    const Dtype* sigmoid_output_data = sigmoid_output_.gpu_data();
    const Dtype* target = bottom[1]->gpu_data();
    Dtype* bottom_diff = bottom[0]->mutable_gpu_diff();
    // Since this memory is never used for anything else,
    // we use to to avoid allocating new GPU memory.
    Dtype* counts = sigmoid_output_.mutable_gpu_diff();
    
    SigmoidCrossEntropyLossBackwardGPU<Dtype><<<CAFFE_GET_BLOCKS(nthreads),
        CAFFE_CUDA_NUM_THREADS>>>(nthreads, sigmoid_output_data, target, bottom_diff,
        has_ignore_label_, ignore_label_, counts);
    CUDA_POST_KERNEL_CHECK;
    
    
    Dtype valid_count = -1;
    // Only launch another CUDA kernel if we actually need the count of valid
    // outputs.
    if (normalization_ == LossParameter_NormalizationMode_VALID &&
        has_ignore_label_) {
      caffe_gpu_asum(nthreads, counts, &valid_count);
    }
    // Scale down gradient
    const Dtype loss_weight = top[0]->cpu_diff()[0];
    caffe_gpu_scal(nthreads, loss_weight / get_normalizer(normalization_, valid_count), bottom_diff);
  }
}

INSTANTIATE_LAYER_GPU_BACKWARD(SigmoidCrossEntropyLossLayer);


}  // namespace caffe
