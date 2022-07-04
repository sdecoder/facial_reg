import cv2
import torch
import torch.utils.data
import os
from torch import nn, optim
from torch.nn import functional as F
from torchvision import datasets, transforms
from structures import CNN_Encoder
from structures import CNN_Decoder
from data_source import MNIST, EMNIST, FashionMNIST
import sys


class Encoder(object):

  def get_model(self):
    return None
  def get_model_weight_name(self):
    return ""

  def _init_dataset(self):
    pass
  def loss_function(self, recon_x, x):
    pass

  def train(self, epoch):
    pass
  def test(self, epoch):
    pass

  def save_to_weight_file(self, path):

    model_name = self.get_model_weight_name()
    model = self.get_model()

    fullpath = '/'.join([path, model_name])
    print(f"[trace] save model to {fullpath}")
    torch.save(model.state_dict(), fullpath)
    pass

  def restore_from_weight_file(self, path):

    self.model_name = self.get_model_weight_name()
    self.model = self.get_model()

    fullpath = '/'.join([path, 'weights', self.model_name])
    print(f"[trace] restore model from {fullpath}")
    if (os.path.exists(fullpath) == False):
      print(f"[trace] weight file {fullpath} does not exist")
      raise FileNotFoundError()

    self.model.load_state_dict(torch.load(fullpath))
    self.model.eval()

  def save_to_onnx_file(self, path, device):

    fullpath = '/'.join([path, self.model_onnx_name])
    channel = 1
    batch_size = 1
    input_shape = (channel, self.image_width, self.image_height)
    model_input = torch.randn(batch_size, *input_shape, requires_grad=True)
    model_input = model_input.to(device)
    torch.onnx.export(self.model,  # model being run
                      model_input,  # model input (or a tuple for multiple inputs)
                      fullpath,  # where to save the model (can be a file or file-like object)
                      export_params=True,  # store the trained parameter weights inside the model file
                      opset_version=10,  # the ONNX version to export the model to
                      do_constant_folding=True,  # whether to execute constant folding for optimization
                      input_names=['input'],  # the model's input names
                      output_names=['output'],  # the model's output names
                      dynamic_axes={'input': {0: 'batch_size'},  # variable length axes
                                    'output': {0: 'batch_size'}})

    pass


  def infer(self, path, device):
    img = cv2.imread(path, cv2.IMREAD_COLOR)
    dim = (self.image_width, self.image_height)    # resize image
    resized_image = cv2.resize(img, dim, interpolation=cv2.INTER_AREA)
    grayImage = cv2.cvtColor(resized_image, cv2.COLOR_RGB2GRAY)
    #cv2.imshow("image", grayImage)
    #cv2.waitKey(0)
    #z = self.encode(x.view(-1, 784))
    grayImageTensor = torch.from_numpy(grayImage)
    grayImageTensor = grayImageTensor.float()
    grayImageTensor = grayImageTensor.view(-1, self.image_width * self.image_height)
    grayImageTensor = grayImageTensor.to(device)
    result = self.model.encode(grayImageTensor)
    if result.is_cuda:
      result = result.cpu()

    return result
