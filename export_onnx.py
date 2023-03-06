import torch
from efficientnet_pytorch import EfficientNet
from torchsummary import summary
model = EfficientNet.from_pretrained('efficientnet-b3').cuda()
dummy_input = torch.randn(1,3,320,320, device='cuda')
summary(model, input_size=(3,320,320),device='cuda')
model.set_swish(memory_efficient=False)
torch.onnx.export(model, dummy_input, "test-b3-2.onnx", verbose=True)
