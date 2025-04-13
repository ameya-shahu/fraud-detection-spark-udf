from onnx_inference import ONNXModel

model = ONNXModel("model.onnx")
pred = model.predict("image_name.png")  # image name
print("Predicted class:", pred)
