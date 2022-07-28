import os
import sys
import logging
import argparse

import numpy as np
from pycuda.compiler import SourceModule

import tensorrt as trt
import pycuda.driver as cuda
import pycuda.autoinit

import numpy

import onnx_graphsurgeon as gs
import argparse
import onnx
import numpy as np
import torch
from PIL import ImageDraw

import common

TRT_LOGGER = trt.Logger()


def get_engine(onnx_file_path, engine_file_path=""):
  """Attempts to load a serialized engine if available, otherwise builds a new TensorRT engine and saves it."""

  def build_engine():
    """Takes an ONNX file and creates a TensorRT engine to run inference with"""
    with trt.Builder(TRT_LOGGER) as builder, builder.create_network(common.EXPLICIT_BATCH) as network, \
        builder.create_builder_config() as config, trt.OnnxParser(network, TRT_LOGGER) as parser, trt.Runtime(
      TRT_LOGGER) as runtime:

      config.max_workspace_size = 1 << 28  # 256MiB
      builder.max_batch_size = 1
      # Parse model file
      if not os.path.exists(onnx_file_path):
        print("ONNX file {} not found, exit -1".format(onnx_file_path))
        exit(-1)

      print("Loading ONNX file from path {}...".format(onnx_file_path))
      with open(onnx_file_path, "rb") as model:
        print("Beginning ONNX file parsing")
        if not parser.parse(model.read()):
          print("ERROR: Failed to parse the ONNX file.")
          for error in range(parser.num_errors):
            print(parser.get_error(error))
          return None
        print("Completed parsing of ONNX file")

      # The actual yolov3.onnx is generated with batch size 64. Reshape input to batch size 1
      print("Building an engine from file {}; this may take a while...".format(onnx_file_path))
      input_batch_size = 1
      input_channel = 3
      input_image_width = 112
      input_image_height = 112
      network.get_input(0).shape = [input_batch_size, input_channel, input_image_width, input_image_height]
      plan = builder.build_serialized_network(network, config)
      if plan == None:
        print("[trace] builder.build_serialized_network failed, exit -1")
        exit(-1)
      engine = runtime.deserialize_cuda_engine(plan)
      print("Completed creating Engine")
      with open(engine_file_path, "wb") as f:
        f.write(plan)
      return engine

  if os.path.exists(engine_file_path):
    # If a serialized engine exists, use it instead of building an engine.
    print("Reading engine from file {}".format(engine_file_path))
    with open(engine_file_path, "rb") as f, trt.Runtime(TRT_LOGGER) as runtime:
      return runtime.deserialize_cuda_engine(f.read())
  else:
    return build_engine()


def main():
  print("[trace] reach func@main")
  onnx_model_path = '../trained/onnx/arcface-resnet100.onnx'
  engine_file_path = '../trained/engine/arcface-resnet100.engine'
  print("[trace] start to build TensorRT engine")
  get_engine(onnx_model_path, engine_file_path)
  pass


if __name__ == "__main__":
  main()
  pass


def pycuda_playground():
  a = numpy.random.randn(4, 4)
  a = a.astype(numpy.float32)
  a_gpu = cuda.mem_alloc(a.nbytes)
  cuda.memcpy_htod(a_gpu, a)

  mod = SourceModule("""
    __global__ void doublify(float *a)
    {
      int idx = threadIdx.x + threadIdx.y*4;
      a[idx] *= 2;
    }
    """)

  func = mod.get_function("doublify")
  func(a_gpu, block=(4, 4, 1))

  a_doubled = numpy.empty_like(a)
  cuda.memcpy_dtoh(a_doubled, a_gpu)
  print(a_doubled)
  print(a)
