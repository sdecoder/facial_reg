import cv2
import torch
import torch.utils.data
from torch import nn, optim
from torch.fx.experimental.fx2trt import to_numpy
from torch.nn import functional as F
from torchvision import datasets, transforms
import time
import os
import sys
sys.path.append('../')

from structures import CNN_Encoder
from structures import CNN_Decoder
from data_source import MNIST, EMNIST, FashionMNIST
import onnx
import onnxruntime

def onnx_infer(onnx_file_path, image_file_path):
  ort_session = onnxruntime.InferenceSession(onnx_file_path, providers=['CPUExecutionProvider'])
  inputs = ort_session.get_inputs()
  #ort_inputs = {ort_session.get_inputs()[0].name: to_numpy(x)}
  #ort_outs = ort_session.run(None, ort_inputs)

  image_width = 28
  image_height = 28
  img = cv2.imread(image_file_path, cv2.IMREAD_COLOR)
  dim = (image_width, image_height)  # resize image
  resized_image = cv2.resize(img, dim, interpolation=cv2.INTER_AREA)
  grayImage = cv2.cvtColor(resized_image, cv2.COLOR_RGB2GRAY)
  # cv2.imshow("image", grayImage)
  # cv2.waitKey(0)
  # z = self.encode(x.view(-1, 784))
  grayImageTensor = torch.from_numpy(grayImage)
  grayImageTensor = grayImageTensor.float()
  grayImageTensor = grayImageTensor.view(1, 1, image_width, image_height)

  print('[trace] prepare the input pair')
  input_pair = {inputs[0].name : to_numpy(grayImageTensor)}
  print('[trace] ort_session.run(None, input_pair)')
  inference_start_time_ns = time.time_ns()
  ort_outs = ort_session.run(None, input_pair)
  inference_end_time_ns = time.time_ns()

  print("[trace] CPUExecutionProvider inference time: {} ns".format(
    int(round((inference_end_time_ns - inference_start_time_ns)))))


  print('[trace] ort_session.run done, showing the ort_outs')
  print(f'[trace] type of ort_outs: {type(ort_outs)}')
  print(f'[trace] ort_outs: {ort_outs}')
  pass

def main():
  onnx_file_path = '../trained/onnx/auto_encoder_encoder_part.onnx'
  if (os.path.exists(onnx_file_path) == False):
    print(f'[trace] specified ONNX file {onnx_file_path} not found')
    exit(-1)

  print('[trace] start to load the onnx file')
  onnx_model = onnx.load(onnx_file_path)
  onnx.checker.check_model(onnx_model)

  print('[trace] onnx.checker.check_model: pass')

  image_file_path = '../results/one.png'
  if (os.path.exists(onnx_file_path) == False):
    print(f'[trace] specified input file {image_file_path} not found')
    exit(-1)

  onnx_infer(onnx_file_path, image_file_path)
  pass

if __name__ == "__main__":
  main()
