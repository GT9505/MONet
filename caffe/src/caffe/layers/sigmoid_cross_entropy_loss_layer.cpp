#include <vector>

#include "caffe/layers/sigmoid_cross_entropy_loss_layer.hpp"
#include "caffe/util/math_functions.hpp"

namespace caffe {

template <typename Dtype>
void SigmoidCrossEntropyLossLayer<Dtype>::LayerSetUp(
    const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
  LossLayer<Dtype>::LayerSetUp(bottom, top);
  sigmoid_bottom_vec_.clear();
  sigmoid_bottom_vec_.push_back(bottom[0]);
  sigmoid_top_vec_.clear();
  sigmoid_top_vec_.push_back(&sigmoid_output_);
  sigmoid_layer_->SetUp(sigmoid_bottom_vec_, sigmoid_top_vec_);
  
  has_ignore_label_ =
    this->layer_param_.loss_param().has_ignore_label();
  if (has_ignore_label_) {
    ignore_label_ = this->layer_param_.loss_param().ignore_label();
  }
  if (!this->layer_param_.loss_param().has_normalization() &&
      this->layer_param_.loss_param().has_normalize()) {
    normalization_ = this->layer_param_.loss_param().normalize() ?
                     LossParameter_NormalizationMode_VALID :
                     LossParameter_NormalizationMode_BATCH_SIZE;
  } else {
    normalization_ = this->layer_param_.loss_param().normalization();
  }
}

template <typename Dtype>
void SigmoidCrossEntropyLossLayer<Dtype>::Reshape(
    const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
  LossLayer<Dtype>::Reshape(bottom, top);
  CHECK_EQ(bottom[0]->count(), bottom[1]->count()) <<
      "SIGMOID_CROSS_ENTROPY_LOSS layer inputs must have the same count.";
  sigmoid_layer_->Reshape(sigmoid_bottom_vec_, sigmoid_top_vec_);
  num_ = bottom[0]->num();
  count_ = bottom[0]->count();
}

template <typename Dtype>
Dtype SigmoidCrossEntropyLossLayer<Dtype>::get_normalizer(
    LossParameter_NormalizationMode normalization_mode, int valid_count) {
  Dtype normalizer;
  switch (normalization_mode) {
    case LossParameter_NormalizationMode_FULL:
      normalizer = Dtype(count_);
      break;
    case LossParameter_NormalizationMode_VALID:
      if (valid_count == -1) {
        normalizer = Dtype(count_);
      } else {
        normalizer = Dtype(valid_count);
      }
      break;
    case LossParameter_NormalizationMode_BATCH_SIZE:
      normalizer = Dtype(num_);
      break;
    case LossParameter_NormalizationMode_NONE:
      normalizer = Dtype(1);
      break;
    default:
      LOG(FATAL) << "Unknown normalization mode: "
          << LossParameter_NormalizationMode_Name(normalization_mode);
  }
  // Some users will have no labels for some examples in order to 'turn off' a
  // particular loss in a multi-task setup. The max prevents NaNs in that case.
  return std::max(Dtype(1.0), normalizer);
}

template <typename Dtype>
void SigmoidCrossEntropyLossLayer<Dtype>::Forward_cpu(
    const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
  // The forward pass computes the sigmoid outputs.
  sigmoid_bottom_vec_[0] = bottom[0];
  sigmoid_layer_->Forward(sigmoid_bottom_vec_, sigmoid_top_vec_);
  // Compute the loss (negative log likelihood)
  const int count = bottom[0]->count();
  // Stable version of loss computation from input data
  const Dtype* input_data = bottom[0]->cpu_data();
  const Dtype* target = bottom[1]->cpu_data();
  Dtype loss = 0;
  int valid_count = 0;
  for (int i = 0; i < count; ++i) {
    if (has_ignore_label_ && target[i] == ignore_label_) {
        continue;
      }
    loss -= input_data[i] * (target[i] - (input_data[i] >= 0)) -
        log(1 + exp(input_data[i] - 2 * input_data[i] * (input_data[i] >= 0)));
    ++valid_count;
  }
  top[0]->mutable_cpu_data()[0] = loss / get_normalizer(normalization_, valid_count);
  
}

template <typename Dtype>
void SigmoidCrossEntropyLossLayer<Dtype>::Backward_cpu(
    const vector<Blob<Dtype>*>& top, const vector<bool>& propagate_down,
    const vector<Blob<Dtype>*>& bottom) {
  if (propagate_down[1]) {
    LOG(FATAL) << this->type()
               << " Layer cannot backpropagate to label inputs.";
  }
  if (propagate_down[0]) {
    // First, compute the diff
    const int count = bottom[0]->count();
    const Dtype* sigmoid_output_data = sigmoid_output_.cpu_data();
    const Dtype* target = bottom[1]->cpu_data();
    Dtype* bottom_diff = bottom[0]->mutable_cpu_diff();
    int valid_count = 0;
    for (int i=0; i<count; ++i) {
      if (has_ignore_label_ && target[i] == ignore_label_) {
        bottom_diff[i] = 0;
      } else {
        bottom_diff[i] = sigmoid_output_data[i] - target[i];
        ++valid_count;
      }
    }
    // Scale down gradient
    const Dtype loss_weight = top[0]->cpu_diff()[0];
    caffe_scal(count, loss_weight / get_normalizer(normalization_, valid_count), bottom_diff);
  }
}

#ifdef CPU_ONLY
STUB_GPU_BACKWARD(SigmoidCrossEntropyLossLayer, Backward);
#endif

INSTANTIATE_CLASS(SigmoidCrossEntropyLossLayer);
REGISTER_LAYER_CLASS(SigmoidCrossEntropyLoss);

}  // namespace caffe
