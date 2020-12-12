import torch
import torchvision

dummy_in = torch.randn(10, 3, 224, 224)
model = torchvision.models.resnet18(pretrained=True)

in_names = [ "actual_input_1" ] + [ "learned_%d" % i for i in range(16) ]
out_names = [ "output1" ]

torch.onnx.export(model, dummy_in, "pytorch.onnx", input_names=in_names, output_names=out_names, opset_version=7, verbose=True)

#pytorch offers native support for onnx, and makes the process of converting simple with one line of code.