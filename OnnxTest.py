import onnx

# Preprocessing: load the ONNX model
model_path = 'pytorch.onnx'
onnx_model = onnx.load(model_path) #loading in the pytorch Model.
model_path = 'model.onnx'
onnx_model2 = onnx.load(model_path)# loading in the tensor flow model.


# Check the model, if an onnx model is not found function will result in error, if one is found will return none.
onnx.checker.check_model(onnx_model)

onnx.checker.check_model(onnx_model2)

print('The models are in onnx format!')