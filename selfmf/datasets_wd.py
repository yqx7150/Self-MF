# coding=utf-8
# Copyright 2020 The Google Research Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# pylint: skip-file
"""Return training and evaluation/test datasets from config files."""
import jax
import tensorflow as tf
import tensorflow_datasets as tfds
import matplotlib.pyplot as plt
import cv2
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from scipy.io import loadmat
import numpy as np
import os

class GetMRI(Dataset):
    def __init__(self,root,augment=None):
        super().__init__()
        self.data_names = np.array([os.path.join(root, x) for x in sorted(os.listdir(root)) if x.lower().endswith(".mat")])
        self.augment = None
        print(self.data_names)

    def k2wgt(self,X,W):
        result = np.multiply(X,W) 
        return result
    @staticmethod
    def generate_1d_mask_with_retention(length, retention_ratio, acs_width, seed=None):
      if seed is not None:
          np.random.seed(seed)
      mask = np.zeros(length, dtype=np.uint8)
      acs_start = (length - acs_width) // 2
      acs_end = acs_start + acs_width
      mask[acs_start:acs_end] = 1
      total_keep = int(length * retention_ratio)
      already_kept = acs_width
      num_random = total_keep - already_kept
      if num_random > 0:
          candidates = np.setdiff1d(np.arange(length), np.arange(acs_start, acs_end))
          sampled = np.random.choice(candidates, size=num_random, replace=False)
          mask[sampled] = 1
      return mask

    @staticmethod
    def expand_to_2d(mask_1d, height):
        return np.tile(mask_1d[np.newaxis, :], (height, 1))
    def __getitem__(self,index):
        # siat_input=loadmat(self.data_names[index])['Img2']
        # siat=np.array(siat_input[:,:,0:2],dtype=np.float32)
        T1_kdata=loadmat(self.data_names[index])['T1']
        T2_kdata=loadmat(self.data_names[index])['T2']
        if T1_kdata.shape != (256, 256) or T2_kdata.shape != (256, 256):
            raise ValueError("Expected 256x256 complex T1 and T2 k-space arrays in the MAT file")
        # siat_complex = siat[:,:,0]+1j*siat[:,:,1]
        # siat_kdata = np.fft.fft2(siat_complex)
        # siat_kdata = np.fft.fftshift(siat_kdata)
        # print(index,T1_kdata.shape,np.max(T1_kdata))
        weight=loadmat(os.environ.get('SELFMF_WEIGHT_MAT', './data/weight.mat'))['weight'] 
        
        mask_1d = self.generate_1d_mask_with_retention(256, retention_ratio=0.75, acs_width=20, seed=0)
        mask_2d = self.expand_to_2d(mask_1d, 256).astype(bool) 
        mask_data = mask_2d if T1_kdata.ndim == 2 else mask_2d[..., np.newaxis]
        masked_T1 = T1_kdata * mask_data
        masked_T2 = T2_kdata * mask_data
        T1_kdata_w = self.k2wgt(masked_T1,weight)  
        T2_kdata_w = self.k2wgt(masked_T2,weight)  #complex
        # print(T1_kdata_w.shape,np.max(T1_kdata_w))
        
        temp = np.zeros((256, 256, 6))
        mask30=loadmat(os.environ.get('SELFMF_MASK30_MAT', './data/mask30.mat'))['mask']
        mask40=loadmat(os.environ.get('SELFMF_MASK40_MAT', './data/mask40.mat'))['mask']

        chan1=np.multiply(T1_kdata_w,mask30)+np.multiply(T2_kdata_w,1-mask30)
        chan2=np.multiply(T1_kdata_w,mask40)+np.multiply(T2_kdata_w,1-mask40)
        kdata = np.array(temp,dtype=np.float32)   
        kdata[:,:,0] = np.real(chan1)
        kdata[:,:,1] = np.imag(chan1)
        kdata[:,:,2] = np.real(chan2)
        kdata[:,:,3] = np.imag(chan2)
        kdata[:,:,4] = np.real(T2_kdata_w)
        kdata[:,:,5] = np.imag(T2_kdata_w)
                
        kdata=kdata.transpose((2,0,1)) # （6，256，256）

        return kdata
    
    def __len__(self):
        return len(self.data_names)

def get_data_scaler(config):
  """Data normalizer. Assume data are always in [0, 1]."""
  if config.data.centered:
    # Rescale to [-1, 1]
    return lambda x: x * 2. - 1.
  else:
    return lambda x: x


def get_data_inverse_scaler(config):
  """Inverse data normalizer."""
  if config.data.centered:
    # Rescale [-1, 1] to [0, 1]
    return lambda x: (x + 1.) / 2.
  else:
    return lambda x: x


def crop_resize(image, resolution):
  """Crop and resize an image to the given resolution."""
  crop = tf.minimum(tf.shape(image)[0], tf.shape(image)[1])
  h, w = tf.shape(image)[0], tf.shape(image)[1]
  image = image[(h - crop) // 2:(h + crop) // 2,
          (w - crop) // 2:(w + crop) // 2]
  image = tf.image.resize(
    image,
    size=(resolution, resolution),
    antialias=True,
    method=tf.image.ResizeMethod.BICUBIC)
  return tf.cast(image, tf.uint8)


def resize_small(image, resolution):
  """Shrink an image to the given resolution."""
  h, w = image.shape[0], image.shape[1]
  ratio = resolution / min(h, w)
  h = tf.round(h * ratio, tf.int32)
  w = tf.round(w * ratio, tf.int32)
  return tf.image.resize(image, [h, w], antialias=True)


def central_crop(image, size):
  """Crop the center of an image to the given size."""
  top = (image.shape[0] - size) // 2
  left = (image.shape[1] - size) // 2
  return tf.image.crop_to_bounding_box(image, top, left, size, size)


def get_dataset(config, uniform_dequantization=False, evaluation=False):
  """Create data loaders for training and evaluation.

  Args:
    config: A ml_collection.ConfigDict parsed from config files.
    uniform_dequantization: If `True`, add uniform dequantization to images.
    evaluation: If `True`, fix number of epochs to 1.

  Returns:
    train_ds, eval_ds, dataset_builder.
  """
  # Compute batch size for this worker.
  batch_size = config.training.batch_size if not evaluation else config.eval.batch_size
  if batch_size % jax.device_count() != 0:
    raise ValueError(f'Batch sizes ({batch_size} must be divided by'
                     f'the number of devices ({jax.device_count()})')

  # Reduce this when image resolution is too large and data pointer is stored
  shuffle_buffer_size = 10000
  prefetch_size = tf.data.experimental.AUTOTUNE
  num_epochs = None if not evaluation else 1

  # Create dataset builders for each dataset.
  if config.data.dataset == 'CIFAR10':
    dataset_builder = tfds.builder('cifar10')
    train_split_name = 'train'
    eval_split_name = 'test'

    def resize_op(img):
      img = tf.image.convert_image_dtype(img, tf.float32)
      return tf.image.resize(img, [config.data.image_size, config.data.image_size], antialias=True)

  elif config.data.dataset == 'SVHN':
    dataset_builder = tfds.builder('svhn_cropped')
    train_split_name = 'train'
    eval_split_name = 'test'

    def resize_op(img):
      img = tf.image.convert_image_dtype(img, tf.float32)
      return tf.image.resize(img, [config.data.image_size, config.data.image_size], antialias=True)

  elif config.data.dataset == 'CELEBA':
    #dataset_builder = tfds.builder('celeb_a')
    train_split_name = 'train'
    eval_split_name = 'validation'

    def resize_op(img):
      img = tf.image.convert_image_dtype(img, tf.float32)
      img = central_crop(img, 140)
      img = resize_small(img, config.data.image_size)
      return img

  #elif config.data.dataset == 'LSUN':
  #  dataset_builder = tfds.builder(f'lsun/{config.data.category}')
  #  train_split_name = 'train'
  #  eval_split_name = 'validation'

  #  if config.data.image_size == 128:
  #    def resize_op(img):
  #      img = tf.image.convert_image_dtype(img, tf.float32)
  #      img = resize_small(img, config.data.image_size)
  #      img = central_crop(img, config.data.image_size)
  #      return img

  #  else:
  #    def resize_op(img):
  #      img = crop_resize(img, config.data.image_size)
  #      img = tf.image.convert_image_dtype(img, tf.float32)
  #      return img
        
  elif config.data.dataset == 'LSUN':
    #dataset_builder = tfds.builder(f'lsun/{config.data.category}')
    train_split_name = 'train'
    eval_split_name = 'validation'

    def resize_op(img):
      img = crop_resize(img, config.data.image_size)
      img = tf.image.convert_image_dtype(img, tf.float32)
      return img

  elif config.data.dataset in ['FFHQ', 'CelebAHQ']:
    dataset_builder = tf.data.TFRecordDataset(config.data.tfrecords_path)
    train_split_name = eval_split_name = 'train'

  else:
    raise NotImplementedError(
      f'Dataset {config.data.dataset} not yet supported.')

  # Customize preprocess functions for each dataset.
  if config.data.dataset in ['FFHQ', 'CelebAHQ']:
    def preprocess_fn(d):
      sample = tf.io.parse_single_example(d, features={
        'shape': tf.io.FixedLenFeature([3], tf.int64),
        'data': tf.io.FixedLenFeature([], tf.string)})
      data = tf.io.decode_raw(sample['data'], tf.uint8)
      data = tf.reshape(data, sample['shape'])
      data = tf.transpose(data, (1, 2, 0))
      img = tf.image.convert_image_dtype(data, tf.float32)
      if config.data.random_flip and not evaluation:
        img = tf.image.random_flip_left_right(img)
      if uniform_dequantization:
        img = (tf.random.uniform(img.shape, dtype=tf.float32) + img * 255.) / 256.
      return dict(image=img, label=None)

  else:
    def preprocess_fn(d):
      """Basic preprocessing function scales data to [0, 1) and randomly flips."""
      img = resize_op(d['image'])
      if config.data.random_flip and not evaluation:
        img = tf.image.random_flip_left_right(img)
      if uniform_dequantization:
        img = (tf.random.uniform(img.shape, dtype=tf.float32) + img * 255.) / 256.

      return dict(image=img, label=d.get('label', None))

  def create_dataset(dataset_builder, split):
    dataset_options = tf.data.Options()
    dataset_options.experimental_optimization.map_parallelization = True
    dataset_options.experimental_threading.private_threadpool_size = 48
    dataset_options.experimental_threading.max_intra_op_parallelism = 1
    read_config = tfds.ReadConfig(options=dataset_options)
    if isinstance(dataset_builder, tfds.core.DatasetBuilder):
      dataset_builder.download_and_prepare()
      ds = dataset_builder.as_dataset(
        split=split, shuffle_files=True, read_config=read_config)
    else:
      ds = dataset_builder.with_options(dataset_options)
    ds = ds.repeat(count=num_epochs)
    ds = ds.shuffle(shuffle_buffer_size)
    ds = ds.map(preprocess_fn, num_parallel_calls=tf.data.experimental.AUTOTUNE)
    ds = ds.batch(batch_size, drop_remainder=True)
    return ds.prefetch(prefetch_size)

  #train_ds = create_dataset(dataset_builder, train_split_name)
  #eval_ds = create_dataset(dataset_builder, eval_split_name)
  
  train_root = os.environ.get("SELFMF_DATA_ROOT", "./data/train")
  eval_root = os.environ.get("SELFMF_EVAL_DATA_ROOT", "./data/eval")
  dataset = GetMRI(root=train_root)
  test_dataset = GetMRI(root=eval_root)
  
  
  train_ds = DataLoader(dataset, batch_size=config.training.batch_size, shuffle=True,
                                num_workers=4)
                                
  eval_ds = DataLoader(test_dataset, batch_size=config.eval.batch_size, shuffle=True,
                                 num_workers=4, drop_last=True)
  
  
  
  
  return train_ds, eval_ds #, dataset_builder
