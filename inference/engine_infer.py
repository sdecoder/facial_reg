import tensorrt as trt
from PIL import Image
import pycuda.driver as cuda
import pycuda.autoinit
import numpy as np
import time
import os
import common
import cv2
import torch
import argparse

# common variables
TRT_LOGGER = trt.Logger(trt.Logger.WARNING)

parser = argparse.ArgumentParser(
  description='Main function to call training for different AutoEncoders')
parser.add_argument('--batch-size', type=int, default=128, metavar='N',
                    help='input batch size for training (default: 128)')
parser.add_argument('--epochs', type=int, default=10, metavar='N',
                    help='number of epochs to train (default: 10)')
parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='enables CUDA training')
parser.add_argument('--seed', type=int, default=42, metavar='S',
                    help='random seed (default: 1)')
parser.add_argument('--log-interval', type=int, default=10, metavar='N',
                    help='how many batches to wait before logging training status')
parser.add_argument('--embedding-size', type=int, default=32, metavar='N',
                    help='how many batches to wait before logging training status')
parser.add_argument('--results_path', type=str, default='results/', metavar='N',
                    help='Where to store images')
parser.add_argument('--models', type=str, default='AE', metavar='N',
                    help='Which architecture to use')
parser.add_argument('--dataset', type=str, default='MNIST', metavar='N',
                    help='Which dataset to use')


def main():
  print("[trace] reach the main entry")
  args = parser.parse_args()
  args.cuda = not args.no_cuda and torch.cuda.is_available()
  device = torch.device("cuda" if args.cuda else "cpu")

  input_image_path = '../results/AlbertEinstein.jpeg'
  if (os.path.exists(input_image_path) == False):
    print(f'[trace] target file {input_image_path} does not exist, exit')
    exit(-1)

  '''
  img = cv2.imread(input_image_path, cv2.IMREAD_COLOR)
  img_width = 28
  img_height = 28
  dim = (img_width, img_height)  # resize image
  resized_image = cv2.resize(img, dim, interpolation=cv2.INTER_AREA)
  grayImage = cv2.cvtColor(resized_image, cv2.COLOR_RGB2GRAY)

  grayImageTensor = torch.from_numpy(grayImage)
  grayImageTensor = grayImageTensor.float()
  grayImageTensor = grayImageTensor.view(-1, img_width * img_height)
  grayImageTensor = grayImageTensor.to(device)
  '''

  #trt_engine_path = '../trained/engine/auto_encoder.engine'
  trt_engine_path = '../trained/engine/arcface-resnet100.engine'

  if (os.path.exists(trt_engine_path) == False):
    print(f'[trace] engine file {trt_engine_path} does not exist, exit')
    exit(-1)

  print('[trace] initiating TensorRT object')
  trtObject = TRTInference(trt_engine_path)
  result = trtObject.infer(input_image_path)
  print(f'[trace] the result type: {type(result[0])}')
  print(f'[trace] the result length: {len(result[0])}')
  print(f'[trace] the result: {result[0]}')
  print('[trace] initilization done')

  pass

# This function is generalized for multiple inputs/outputs.
# inputs and outputs are expected to be lists of HostDeviceMem objects.
def do_inference(context, bindings, inputs, outputs, stream, batch_size=1):
  print('[trace] reach the do_inference')
  # Transfer input data to the GPU.
  print('[trace] cuda.memcpy_htod_async')
  [cuda.memcpy_htod_async(inp.device, inp.host, stream) for inp in inputs]
  # Run inference.
  print('[trace] context.execute_async')
  context.execute_async(batch_size=batch_size, bindings=bindings, stream_handle=stream.handle)
  # Transfer predictions back from the GPU.

  print('[trace] cuda.memcpy_dtoh_async')
  [cuda.memcpy_dtoh_async(out.host, out.device, stream) for out in outputs]
  # Synchronize the stream

  print('[trace] stream.synchronize()')
  stream.synchronize()
  # Return only the host outputs.
  return [out.host for out in outputs]


class TRTInference(object):
  """Manages TensorRT objects for model inference."""

  def __init__(self, trt_engine_path, trt_engine_datatype=trt.DataType.FLOAT, calib_dataset=None, batch_size=1):
    """Initializes TensorRT objects needed for model inference.
    Args:
        trt_engine_path (str): path where TensorRT engine should be stored
        uff_model_path (str): path of .uff model
        trt_engine_datatype (trt.DataType):
            requested precision of TensorRT engine used for inference
        batch_size (int): batch size for which engine
            should be optimized for
    """

    # We first load all custom plugins shipped with TensorRT,
    # some of them will be needed during inference
    trt.init_libnvinfer_plugins(TRT_LOGGER, '')

    # Initialize runtime needed for loading TensorRT engine from file
    self.trt_runtime = trt.Runtime(TRT_LOGGER)
    # TRT engine placeholder
    self.trt_engine = None

    # Display requested engine settings to stdout
    print("TensorRT inference engine settings:")
    print("  * Inference precision - {}".format(trt_engine_datatype))
    print("  * Max batch size - {}\n".format(batch_size))

    # If we get here, the file with engine exists, so we can load it

    print("[trace] Loading cached TensorRT engine from {}".format(trt_engine_path))
    self.trt_engine = common.load_engine(self.trt_runtime, trt_engine_path)
    # This allocates memory for network inputs/outputs on both CPU and GPU
    print("[trace] TensorRT engine loaded")
    print("[trace] allocating buffers for TensorRT engine")
    self.inputs, self.outputs, self.bindings, self.stream = common.allocate_buffers(self.trt_engine)
    print("[trace] allocating buffers done")
    # Execution context is needed for inference

    print("[trace] TensorRT engine: creating execution context")
    self.context = self.trt_engine.create_execution_context()

    print("[trace] TensorRT engine: execution context created.")
    # Allocate memory for multiple usage [e.g. multiple batch inference]
    batch_size = 1
    channel = 3
    width = 112
    height = width
    INPUT_SHAPE = (channel, width, height)

    input_volume = trt.volume(INPUT_SHAPE)
    print("[trace] TensorRT volume of input: created")

    self.numpy_array = np.zeros((self.trt_engine.max_batch_size, input_volume))

  def infer(self, image_path):
    """Infers model on given image.
    Args:
        image_path (str): image to run object detection model on
    """

    # Load image into CPU
    print(f'[trace] TensorRT inference for input image{image_path}')
    img = self._load_img(image_path)

    # Copy it into appropriate place into memory
    # (self.inputs was returned earlier by allocate_buffers())
    np.copyto(self.inputs[0].host, img.ravel())

    # When infering on single image, we measure inference
    # time to output it to the user
    inference_start_time = time.time()

    # Fetch output from the model
    inference_start_time_ns = time.time_ns()
    retObj = do_inference(
      self.context, bindings=self.bindings, inputs=self.inputs,
      outputs=self.outputs, stream=self.stream)
    inference_end_time_ns = time.time_ns()

    # Output inference time
    print("TensorRT inference time: {} ms".format(
      int(round((time.time() - inference_start_time) * 1000))))
    print("[trace] TensorRT inference time: {} ns".format(
      int(round((inference_end_time_ns - inference_start_time_ns)))))

    return retObj
    # And return results
    #return detection_out, keepCount_out

  def infer_webcam(self, arr):
    """Infers model on given image.
    Args:
        arr (numpy array): image to run object detection model on
    """

    # Load image into CPU
    print('[trace] infer_webcam(self, arr)')
    img = self._load_img_webcam(arr)

    # Copy it into appropriate place into memory
    # (self.inputs was returned earlier by allocate_buffers())
    np.copyto(self.inputs[0].host, img.ravel())

    # When infering on single image, we measure inference
    # time to output it to the user
    inference_start_time = time.time()

    # Fetch output from the model
    [detection_out, keepCount_out] = do_inference(
      self.context, bindings=self.bindings, inputs=self.inputs,
      outputs=self.outputs, stream=self.stream)

    # Output inference time
    print("[trace] TensorRT inference time: {} ms".format(
      int(round((time.time() - inference_start_time) * 1000))))


    # And return results
    return detection_out, keepCount_out

  def infer_batch(self, image_paths):

    print('[trace] infer_batch(self, image_paths):')
    """Infers model on batch of same sized images resized to fit the model.
    Args:
        image_paths (str): paths to images, that will be packed into batch
            and fed into model
    """

    # Verify if the supplied batch size is not too big
    max_batch_size = self.trt_engine.max_batch_size
    actual_batch_size = len(image_paths)
    if actual_batch_size > max_batch_size:
      raise ValueError(
        "image_paths list bigger ({}) than engine max batch size ({})".format(actual_batch_size, max_batch_size))

    # Load all images to CPU...
    imgs = self._load_imgs(image_paths)
    # ...copy them into appropriate place into memory...
    # (self.inputs was returned earlier by allocate_buffers())
    np.copyto(self.inputs[0].host, imgs.ravel())

    # ...fetch model outputs...
    [detection_out, keep_count_out] = do_inference(
      self.context, bindings=self.bindings, inputs=self.inputs,
      outputs=self.outputs, stream=self.stream,
      batch_size=max_batch_size)
    # ...and return results.
    return detection_out, keep_count_out

  def _load_image_into_numpy_array(self, image):
    (im_width, im_height) = image.size
    new_shape = (im_height, im_width, 1)
    return np.array(image).reshape(new_shape).astype(np.uint8)

  def _load_imgs(self, image_paths):
    batch_size = self.trt_engine.max_batch_size
    for idx, image_path in enumerate(image_paths):
      img_np = self._load_img(image_path)
      self.numpy_array[idx] = img_np
    return self.numpy_array

  def _load_img_webcam(self, arr):
    image = Image.fromarray(np.uint8(arr))
    model_input_width = 28
    model_input_height = 28
    # Note: Bilinear interpolation used by Pillow is a little bit
    # different than the one used by Tensorflow, so if network receives
    # an image that is not 300x300, the network output may differ
    # from the one output by Tensorflow
    image_resized = image.resize(
      size=(model_input_width, model_input_height),
      resample=Image.BILINEAR
    )
    img_np = self._load_image_into_numpy_array(image_resized)
    # HWC -> CHW
    img_np = img_np.transpose((2, 0, 1))
    # Normalize to [-1.0, 1.0] interval (expected by model)
    img_np = (2.0 / 255.0) * img_np - 1.0
    img_np = img_np.ravel()
    return img_np

  def _load_img(self, image_path):

    image_width = 112
    image_height = 112
    img = cv2.imread(image_path, cv2.IMREAD_COLOR)
    dim = (image_width, image_height)  # resize im
    resized_image = cv2.resize(img, dim, interpolation=cv2.INTER_AREA)
    resized_image_tensor = torch.from_numpy(resized_image)
    resized_image_tensor = resized_image_tensor.float()
    resized_image_tensor = resized_image_tensor.view(1, -1, image_width, image_height)
    print(f'[trace] resized_image_tensor.size(): {resized_image_tensor.size()}')
    #using arcface-resnet100.engine as the inference engine

    return resized_image_tensor

    image_width = 28
    image_height = 28
    img = cv2.imread(image_path, cv2.IMREAD_COLOR)
    dim = (image_width, image_height)  # resize image
    resized_image = cv2.resize(img, dim, interpolation=cv2.INTER_AREA)
    grayImage = cv2.cvtColor(resized_image, cv2.COLOR_RGB2GRAY)
    grayImageTensor = torch.from_numpy(grayImage)
    grayImageTensor = grayImageTensor.float()
    grayImageTensor = grayImageTensor.view(1, 1, image_width, image_height)

    return grayImageTensor

    print(f'[trace] _load_img for input path {image_path}')
    image = Image.open(image_path)
    model_input_width = 28
    model_input_height = 28
    # Note: Bilinear interpolation used by Pillow is a little bit
    # different than the one used by Tensorflow, so if network receives
    # an image that is not 300x300, the network output may differ
    # from the one output by Tensorflow
    image_resized = image.resize(
      size=(model_input_width, model_input_height),
      resample=Image.BILINEAR
    )
    img_np = self._load_image_into_numpy_array(image_resized)
    # HWC -> CHW
    #img_np = img_np.transpose((2, 0, 1))
    # Normalize to [-1.0, 1.0] interval (expected by model)
    # img_np = (2.0 / 255.0) * img_np - 1.0
    #img_np = img_np.ravel()
    return img_np


if __name__ == "__main__":
  main()
  pass
