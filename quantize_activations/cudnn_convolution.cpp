#include <torch/extension.h>
#include <vector>
#include <ATen/NativeFunctions.h>
#include <ATen/Config.h>

/*
PyTorch extension enabling direct access to the following cuDNN-accelerated C++ functions
that are included in PyTorch:
    - cudnn_convolution
    - cudnn_convolution_backward_weight
    - cudnn_convolution_backward_input
The functions defined here can be called from Python in replacement of
torch.nn.conv2d, torch.nn.grad.conv2d_weight and torch.nn.grad.conv2d_input,
and run significantly faster. See 'example.py' for how these functions
are called.
Adapted from code posted by hanspinckaers:
https://discuss.pytorch.org/t/cuda-error-with-cudnn-convolution-backward-weight-function/41214
*/
namespace at { namespace native {
std::tuple<Tensor, Tensor, Tensor> my_convolution_backward(
    const Tensor& grad_output_, const Tensor& input_, const Tensor& weight_,
    const at::OptionalIntArrayRef bias_sizes_opt,
    IntArrayRef stride, IntArrayRef padding, IntArrayRef dilation, bool transposed, IntArrayRef output_padding,
    int64_t groups, std::array<bool, 3> output_mask){
        return at::convolution_backward(
            grad_output_, input_, weight_, 
            bias_sizes_opt, stride, padding, dilation, 
            transposed, output_padding, groups, output_mask);
    }
}}
// at::Tensor convolution_backward_input(
//     c10::ArrayRef<int64_t> input_size,
//     const at::Tensor& weight,
//     const at::Tensor& grad_output,
//     c10::ArrayRef<int64_t> stride,
//     c10::ArrayRef<int64_t> padding,
//     c10::ArrayRef<int64_t> dilation,
//     int64_t groups,
//     bool benchmark,
//     bool deterministic,
//     bool allow_tf32) {

//     return at::convolution_backward(
//         input_size,
//         grad_output,
//         weight,
//         padding,
//         stride,
//         dilation,
//         groups,
//         benchmark,
//         deterministic,
//         allow_tf32);
// }

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("convolution_backward_weight", &at::native::my_convolution_backward, "convolution backward weight");
    // m.def("convolution_backward_input", &convolution_backward_input, "convolution backward input");
}