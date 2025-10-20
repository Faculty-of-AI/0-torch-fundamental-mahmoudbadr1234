
import torch


tensor_A = torch.rand(7, 7)
print("Tensor A:\n", tensor_A)



tensor_B = torch.rand(1, 7)
print("\nTensor B:\n", tensor_B)


result = torch.mm(tensor_A, tensor_B.T)
print("\nMatrix Multiplication Result:\n", result)


torch.manual_seed(0)


tensor_A = torch.rand(7, 7)
tensor_B = torch.rand(1, 7)


result = torch.mm(tensor_A, tensor_B.T)

print("\nResult after setting manual seed:\n", result)
print("\nResult shape:", result.shape)
