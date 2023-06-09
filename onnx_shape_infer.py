import onnx
from onnx import shape_inference

input_path = "/home/br/program/plard/plard_sim.onnx"
output_path = "/home/br/program/plard/plard_sim_infer_shape.onnx"
model_str = onnx.load(input_path)
model_infer_shape_str = shape_inference.infer_shapes(model_str)
onnx.save(model_infer_shape_str, output_path)