import torch
import torch.onnx


from ptsemseg.models import get_model


def convert_onnx(model):

  # set the model to inference model
  model.eval()

  # create a dummy input tensor
  dummy_input_img = torch.randn([1, 3, 384, 1280])
  dummy_input_lidar = torch.randn([1, 1, 384, 1280])
  # export the model
  torch.onnx.export(model,      # model being run
      [dummy_input_img, dummy_input_lidar],              # model input (or a tuple for multiple input)
      "/home/br/program/plard/plard.onnx",             # where to save the model
      export_params=True,       # store the trained parameter weights inside the model file
      opset_version=11,         # the ONNX version to export the model to
      do_constant_folding=True, # whether tot execute constant folding for optimization
      input_names=["input_img", "input_lidar"],          # the model's input name
      output_names=["output"],         # the model's output name
      dynamic_axes=None)         # variable length axes  ????

  print(" ")
  print("model has been converted to ONNX")


if __name__ == "__main__":
  model = get_model("plard", 2)
  path = "/home/br/program/plard/plard_kitti_road.pth"
  state = torch.load(path)["model_state"]
  model.load_state_dict(state)
  # test with image

  # convert to onnx
  convert_onnx(model)

