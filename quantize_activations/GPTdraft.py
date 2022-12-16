import torch
# define a new max pooling layer using torch.autograd
class qMaxPool2d(torch.autograd.Function):
  @staticmethod
  def forward(ctx, input, kernel_size, stride):
     # apply the max pooling operation using a built-in function
     output = torch.nn.functional.max_pool2d(input, kernel_size, stride)

     # save the input and the parameters for the backward step
     ctx.save_for_backward(input, kernel_size, stride)

     return output

  @staticmethod
  def backward(ctx, grad_output):
     # retrieve the saved tensors
     input, kernel_size, stride = ctx.saved_tensors

     # apply the max unpooling operation using a built-in function
     grad_input = torch.nn.functional.max_unpool2d(grad_output, input, kernel_size, stride)

     return grad_input, None, None

if __name__ == '__main__':
  # create a MyMaxPool2d layer
  my_max_pool_2d = qMaxPool2d()

  # create a MaxPool2d layer
  max_pool_2d = torch.nn.MaxPool2d(2, 2)

  # create some input data
  input = torch.randn(2, 3, 4, 4)

  # run the input through the MyMaxPool2d and MaxPool2d layers
  my_output = my_max_pool_2d(input, 2, 2)
  output = max_pool_2d(input)

  # print the outputs
  print(my_output)
  print(output)
  