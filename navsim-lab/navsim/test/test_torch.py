import torch

def test_torch_cuda_available(self):
    self.assertTrue(torch.cuda.is_available())


