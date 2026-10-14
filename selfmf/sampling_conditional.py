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
# pytype: skip-file
"""Various sampling methods."""
import functools

import torch
import numpy as np
import abc
import os
from models.utils import from_flattened_numpy, to_flattened_numpy, get_score_fn
from scipy import integrate
import sde_lib
from models import utils as mutils
#from skimage.measure import compare_psnr,compare_ssim
from skimage.metrics import structural_similarity as compare_ssim
from skimage.metrics import peak_signal_noise_ratio as compare_psnr
from skimage.metrics import mean_squared_error as compare_mse
import cv2
import os.path as osp
import matplotlib.pyplot as plt
import scipy.io as io
from SAKE import fft2c, ifft2c, im2row, row2im, sake
from mri_ops import enforce_acquired_samples
import time
def fft2c_noshift(x):
    size = (x).shape
    fctr = size[0]*size[1]
    Kdata = np.zeros((size),dtype=np.complex64)
    for i in range(size[2]):
        Kdata[:,:,i] = (1/np.sqrt(fctr))*np.fft.fft2(x[:,:,i])
    return Kdata
def write_kdata(Kdata,name,path):
    temp = np.log(1+abs(Kdata))    
    plt.axis('off')
    plt.imshow(abs(temp),cmap='gray')
    plt.savefig(osp.join(path,name),transparent=True, dpi=128, pad_inches = 0,bbox_inches = 'tight')

def write_Data(model_num,psnr,ssim,name,path):
    filedir=name+"result.txt"
    with open(osp.join(path,filedir),"w+") as f:#a+
        f.writelines(str(model_num)+' '+'['+str(round(psnr, 2))+' '+str(round(ssim, 4))+']')
        f.write('\n')
        
def write_Data2(psnr,ssim,name,path):
    filedir=name+"PC_SAKE.txt"
    with open(osp.join(path,filedir),"a+") as f:#a+
        f.writelines('['+str(round(psnr, 2))+' '+str(round(ssim, 4))+']')
        f.write('\n')
        
def write_images(x,image_save_path):
    x = np.clip(x * 255, 0, 255).astype(np.uint8)
    cv2.imwrite(image_save_path, x)
  
def KtoImg(k,save_path,coil):
  img = np.zeros((coil,256,256),dtype=np.complex64)
  for i in range(coil):
    img[i,:,:]=np.fft.ifft2(k[:,:,i])
  img_sos = np.sqrt(np.sum(np.square(np.abs(img)),axis=0))
  #plt.imsave(save_path, img_sos, cmap='gray')
  img_sos = np.clip(img_sos * 255, 0, 255).astype(np.uint8)
  cv2.imwrite(save_path, img_sos)

def k2wgt(X,W):
    Y = np.multiply(X,W) 
    return Y

def wgt2k(X,W,DC):
    Y = np.multiply(X,1./W)
    Y[W==0] = DC[W==0] 
    return Y
    
_CORRECTORS = {}
_PREDICTORS = {}


def register_predictor(cls=None, *, name=None):
  """A decorator for registering predictor classes."""

  def _register(cls):
    if name is None:
      local_name = cls.__name__
    else:
      local_name = name
    if local_name in _PREDICTORS:
      raise ValueError(f'Already registered model with name: {local_name}')
    _PREDICTORS[local_name] = cls
    return cls

  if cls is None:
    return _register
  else:
    return _register(cls)


def register_corrector(cls=None, *, name=None):
  """A decorator for registering corrector classes."""

  def _register(cls):
    if name is None:
      local_name = cls.__name__
    else:
      local_name = name
    if local_name in _CORRECTORS:
      raise ValueError(f'Already registered model with name: {local_name}')
    _CORRECTORS[local_name] = cls
    return cls

  if cls is None:
    return _register
  else:
    return _register(cls)


def get_predictor(name):
  return _PREDICTORS[name]


def get_corrector(name):
  return _CORRECTORS[name]


def get_sampling_fn(config, sde, shape, inverse_scaler, eps):
  """Create a sampling function.

  Args:
    config: A `ml_collections.ConfigDict` object that contains all configuration information.
    sde: A `sde_lib.SDE` object that represents the forward SDE.
    shape: A sequence of integers representing the expected shape of a single sample.
    inverse_scaler: The inverse data normalizer function.
    eps: A `float` number. The reverse-time SDE is only integrated to `eps` for numerical stability.

  Returns:
    A function that takes random states and a replicated training state and outputs samples with the
      trailing dimensions matching `shape`.
  """

  sampler_name = config.sampling.method
  # Probability flow ODE sampling with black-box ODE solvers
  if sampler_name.lower() == 'ode':
    sampling_fn = get_ode_sampler(sde=sde,
                                  shape=shape,
                                  inverse_scaler=inverse_scaler,
                                  denoise=config.sampling.noise_removal,
                                  eps=eps,
                                  device=config.device)
  # Predictor-Corrector sampling. Predictor-only and Corrector-only samplers are special cases.
  elif sampler_name.lower() == 'pc':
    predictor = get_predictor(config.sampling.predictor.lower())
    corrector = get_corrector(config.sampling.corrector.lower())
    sampling_fn = get_pc_sampler(sde=sde,
                                 shape=shape,
                                 predictor=predictor,
                                 corrector=corrector,
                                 inverse_scaler=inverse_scaler,
                                 snr=config.sampling.snr,
                                 n_steps=config.sampling.n_steps_each,
                                 probability_flow=config.sampling.probability_flow,
                                 continuous=config.training.continuous,
                                 denoise=config.sampling.noise_removal,
                                 eps=eps,
                                 device=config.device)
  else:
    raise ValueError(f"Sampler name {sampler_name} unknown.")

  return sampling_fn


class Predictor(abc.ABC):
  """The abstract class for a predictor algorithm."""

  def __init__(self, sde, score_fn, probability_flow=False):
    super().__init__()
    self.sde = sde
    # Compute the reverse SDE/ODE
    self.rsde = sde.reverse(score_fn, probability_flow)
    self.score_fn = score_fn

  @abc.abstractmethod
  def update_fn(self, x, t):
    """One update of the predictor.

    Args:
      x: A PyTorch tensor representing the current state
      t: A Pytorch tensor representing the current time step.

    Returns:
      x: A PyTorch tensor of the next state.
      x_mean: A PyTorch tensor. The next state without random noise. Useful for denoising.
    """
    pass


class Corrector(abc.ABC):
  """The abstract class for a corrector algorithm."""

  def __init__(self, sde, score_fn, snr, n_steps):
    super().__init__()
    self.sde = sde
    self.score_fn = score_fn
    self.snr = snr
    self.n_steps = n_steps

  @abc.abstractmethod
  def update_fn(self, x, t):
    """One update of the corrector.

    Args:
      x: A PyTorch tensor representing the current state
      t: A PyTorch tensor representing the current time step.

    Returns:
      x: A PyTorch tensor of the next state.
      x_mean: A PyTorch tensor. The next state without random noise. Useful for denoising.
    """
    pass


@register_predictor(name='euler_maruyama')
class EulerMaruyamaPredictor(Predictor):
  def __init__(self, sde, score_fn, probability_flow=False):
    super().__init__(sde, score_fn, probability_flow)

  def update_fn(self, x, t):
    dt = -1. / self.rsde.N
    z = torch.randn_like(x)
    drift, diffusion = self.rsde.sde(x, t)
    x_mean = x + drift * dt
    x = x_mean + diffusion[:, None, None, None] * np.sqrt(-dt) * z
    return x, x_mean


@register_predictor(name='reverse_diffusion')
class ReverseDiffusionPredictor(Predictor):
  def __init__(self, sde, score_fn, probability_flow=False):
    super().__init__(sde, score_fn, probability_flow)
  
  # Alogrithm 2
  def update_fn(self, x, t):
 
    f, G = self.rsde.discretize(x, t) # 3
    z = torch.randn_like(x) # 4
    x_mean = x - f # 3
    x = x_mean + G[:, None, None, None] * z # 5  
    
    return x, x_mean


@register_predictor(name='ancestral_sampling')
class AncestralSamplingPredictor(Predictor):
  """The ancestral sampling predictor. Currently only supports VE/VP SDEs."""

  def __init__(self, sde, score_fn, probability_flow=False):
    super().__init__(sde, score_fn, probability_flow)
    if not isinstance(sde, sde_lib.VPSDE) and not isinstance(sde, sde_lib.VESDE):
      raise NotImplementedError(f"SDE class {sde.__class__.__name__} not yet supported.")
    assert not probability_flow, "Probability flow not supported by ancestral sampling"

  def vesde_update_fn(self, x, t):
    sde = self.sde
    timestep = (t * (sde.N - 1) / sde.T).long()
    sigma = sde.discrete_sigmas[timestep]
    adjacent_sigma = torch.where(timestep == 0, torch.zeros_like(t), sde.discrete_sigmas.to(t.device)[timestep - 1])
    score = self.score_fn(x, t)
    x_mean = x + score * (sigma ** 2 - adjacent_sigma ** 2)[:, None, None, None]
    std = torch.sqrt((adjacent_sigma ** 2 * (sigma ** 2 - adjacent_sigma ** 2)) / (sigma ** 2))
    noise = torch.randn_like(x)
    x = x_mean + std[:, None, None, None] * noise
    return x, x_mean

  def vpsde_update_fn(self, x, t):
    sde = self.sde
    timestep = (t * (sde.N - 1) / sde.T).long()
    beta = sde.discrete_betas.to(t.device)[timestep]
    score = self.score_fn(x, t)
    x_mean = (x + beta[:, None, None, None] * score) / torch.sqrt(1. - beta)[:, None, None, None]
    noise = torch.randn_like(x)
    x = x_mean + torch.sqrt(beta)[:, None, None, None] * noise
    return x, x_mean

  def update_fn(self, x, t):
    if isinstance(self.sde, sde_lib.VESDE):
      return self.vesde_update_fn(x, t)
    elif isinstance(self.sde, sde_lib.VPSDE):
      return self.vpsde_update_fn(x, t)


@register_predictor(name='none')
class NonePredictor(Predictor):
  """An empty predictor that does nothing."""

  def __init__(self, sde, score_fn, probability_flow=False):
    pass

  def update_fn(self, x, t):
    return x, x


@register_corrector(name='langevin')
class LangevinCorrector(Corrector):
  def __init__(self, sde, score_fn, snr, n_steps):
    super().__init__(sde, score_fn, snr, n_steps)
    if not isinstance(sde, sde_lib.VPSDE) \
        and not isinstance(sde, sde_lib.VESDE) \
        and not isinstance(sde, sde_lib.subVPSDE):
      raise NotImplementedError(f"SDE class {sde.__class__.__name__} not yet supported.")

  def update_fn(self, x1,x2,x3,x_mean,t):
    sde = self.sde
    score_fn = self.score_fn
    n_steps = self.n_steps
    target_snr = self.snr
    if isinstance(sde, sde_lib.VPSDE) or isinstance(sde, sde_lib.subVPSDE):
      timestep = (t * (sde.N - 1) / sde.T).long()
      alpha = sde.alphas.to(t.device)[timestep]
    else:
      alpha = torch.ones_like(t)
    
    # Algorithm 4
    for i in range(n_steps):
   
      grad1 = score_fn(x1, t) # 5 
      grad2 = score_fn(x2, t) # 5 
      grad3 = score_fn(x3, t) # 5 
      
      noise1 = torch.randn_like(x1) # 4 
      noise2 = torch.randn_like(x2) # 4
      noise3 = torch.randn_like(x3) # 4

      
      grad_norm1 = torch.norm(grad1.reshape(grad1.shape[0], -1), dim=-1).mean()
      noise_norm1 = torch.norm(noise1.reshape(noise1.shape[0], -1), dim=-1).mean()
      grad_norm2 = torch.norm(grad2.reshape(grad2.shape[0], -1), dim=-1).mean()
      noise_norm2 = torch.norm(noise2.reshape(noise2.shape[0], -1), dim=-1).mean()      
      grad_norm3 = torch.norm(grad3.reshape(grad3.shape[0], -1), dim=-1).mean()
      noise_norm3 = torch.norm(noise3.reshape(noise3.shape[0], -1), dim=-1).mean()            
      
      grad_norm =(grad_norm1+grad_norm2+grad_norm3)/3.0
      noise_norm = (noise_norm1+noise_norm2+noise_norm3)/3.0
      
      step_size =  (2 * alpha)*((target_snr * noise_norm / grad_norm) ** 2 ) # 6 
   
      x_mean = x_mean + step_size[:, None, None, None] * (grad1+grad2+grad3)/3.0 # 7
      
      x1 = x_mean + torch.sqrt(step_size * 2)[:, None, None, None] * noise1 # 7
      x2 = x_mean + torch.sqrt(step_size * 2)[:, None, None, None] * noise2 # 7
      x3 = x_mean + torch.sqrt(step_size * 2)[:, None, None, None] * noise3 # 7
      
    return x1,x2,x3,x_mean


@register_corrector(name='ald')
class AnnealedLangevinDynamics(Corrector):
  """The original annealed Langevin dynamics predictor in NCSN/NCSNv2.

  We include this corrector only for completeness. It was not directly used in our paper.
  """

  def __init__(self, sde, score_fn, snr, n_steps):
    super().__init__(sde, score_fn, snr, n_steps)
    if not isinstance(sde, sde_lib.VPSDE) \
        and not isinstance(sde, sde_lib.VESDE) \
        and not isinstance(sde, sde_lib.subVPSDE):
      raise NotImplementedError(f"SDE class {sde.__class__.__name__} not yet supported.")

  def update_fn(self, x, t):
    sde = self.sde
    score_fn = self.score_fn
    n_steps = self.n_steps
    target_snr = self.snr
    if isinstance(sde, sde_lib.VPSDE) or isinstance(sde, sde_lib.subVPSDE):
      timestep = (t * (sde.N - 1) / sde.T).long()
      alpha = sde.alphas.to(t.device)[timestep]
    else:
      alpha = torch.ones_like(t)

    std = self.sde.marginal_prob(x, t)[1]
   
    for i in range(n_steps):
      grad = score_fn(x, t)
      noise = torch.randn_like(x)
      step_size = (target_snr * std) ** 2 * 2 * alpha
      x_mean = x + step_size[:, None, None, None] * grad
      x = x_mean + noise * torch.sqrt(step_size * 2)[:, None, None, None]
    return x, x_mean


@register_corrector(name='none')
class NoneCorrector(Corrector):
  """An empty corrector that does nothing."""

  def __init__(self, sde, score_fn, snr, n_steps):
    pass

  def update_fn(self, x, t):
    return x, x


def shared_predictor_update_fn(x, t, sde, model, predictor, probability_flow, continuous):
  """A wrapper that configures and returns the update function of predictors."""
  score_fn = mutils.get_score_fn(sde, model, train=False, continuous=continuous)
  if predictor is None:
    # Corrector-only sampler
    predictor_obj = NonePredictor(sde, score_fn, probability_flow)
  else:
    predictor_obj = predictor(sde, score_fn, probability_flow)
  return predictor_obj.update_fn(x, t)


def shared_corrector_update_fn(x1,x2,x3,x_mean,t,sde, model, corrector, continuous, snr, n_steps):
  """A wrapper tha configures and returns the update function of correctors."""
  score_fn = mutils.get_score_fn(sde, model, train=False, continuous=continuous)
  if corrector is None:
    # Predictor-only sampler
    corrector_obj = NoneCorrector(sde, score_fn, snr, n_steps)
  else:
    corrector_obj = corrector(sde, score_fn, snr, n_steps)
  return corrector_obj.update_fn(x1,x2,x3,x_mean,t)


def get_pc_sampler(sde, shape, predictor, corrector, inverse_scaler, snr,
                   n_steps=1, probability_flow=False, continuous=False,
                   denoise=True, eps=1e-3, device='cuda'):
  """Create a Predictor-Corrector (PC) sampler.

  Args:
    sde: An `sde_lib.SDE` object representing the forward SDE.
    shape: A sequence of integers. The expected shape of a single sample.
    predictor: A subclass of `sampling.Predictor` representing the predictor algorithm.
    corrector: A subclass of `sampling.Corrector` representing the corrector algorithm.
    inverse_scaler: The inverse data normalizer.
    snr: A `float` number. The signal-to-noise ratio for configuring correctors.
    n_steps: An integer. The number of corrector steps per predictor update.
    probability_flow: If `True`, solve the reverse-time probability flow ODE when running the predictor.
    continuous: `True` indicates that the score model was continuously trained.
    denoise: If `True`, add one-step denoising to the final samples.
    eps: A `float` number. The reverse-time SDE and ODE are integrated to `epsilon` to avoid numerical issues.
    device: PyTorch device.

  Returns:
    A sampling function that returns samples and the number of function evaluations during sampling.
  """
  # Create predictor & corrector update functions
  predictor_update_fn = functools.partial(shared_predictor_update_fn,
                                          sde=sde,
                                          predictor=predictor,
                                          probability_flow=probability_flow,
                                          continuous=continuous)
  corrector_update_fn = functools.partial(shared_corrector_update_fn,
                                          sde=sde,
                                          corrector=corrector,
                                          continuous=continuous,
                                          snr=snr,
                                          n_steps=n_steps)

  def pc_sampler(model,test_data,img_name,save_path):
    """ The PC sampler funciton.
    Args:
      model: A score model.
    Returns:
      Samples, number of function evaluations.
    """
    with torch.no_grad():
      
      # Initial sample
      #x = sde.prior_sampling(shape).to(device) # 1
      os.makedirs(save_path, exist_ok=True)
      timesteps = torch.linspace(sde.T, eps, sde.N, device=device)
      
      coil = 12
      
      T1_ori_data = np.zeros([coil,256,256],dtype=np.complex64)
      T1_ori_data_img = np.zeros([256,256,coil],dtype=np.complex64)
      reference_key = os.environ.get('SELFMF_REFERENCE_KEY', 'T1_img')
      T1_ori_data_img = test_data[reference_key]
      T1_ori_data_img = T1_ori_data_img/np.max(abs(T1_ori_data_img))
      for i in range(coil):
        T1_ori_data[i,:,:] = np.fft.fft2(T1_ori_data_img[:,:,i]) # T1 k space
    
      ori_data = np.zeros([256,256,coil],dtype=np.complex64) 
      target_key = os.environ.get('SELFMF_TARGET_KEY', 'T2_img')
      ori_data = test_data[target_key]
      ori_data = ori_data/np.max(abs(ori_data))
      ori = np.copy(ori_data)
      
      ori_data = np.swapaxes(ori_data,0,2)
      ori_data = np.swapaxes(ori_data,1,2) #(coils,256,256)
      ori_data_sos = np.sqrt(np.sum(np.square(np.abs(ori_data)),axis=0)) 
      write_images(abs(ori_data_sos),osp.join(save_path,img_name+'ori'+'.png'))
      
      #======== mask
      mask = np.zeros((coil,256,256))
      #mask_path = './parallel_inputdata/mask/radial/mask_radial_015.mat'
      mask_path = os.environ.get('SELFMF_SAMPLING_MASK', './data/sampling_mask.mat')
      # ./parallel_inputdata/mask/radial/mask_radial_015.mat  (R6)
      # mask_item = io.loadmat(mask_path)['mask']
      # mask_item = io.loadmat('./parallel_inputdata/mask/random2D/6.mat')['mask']
      mask_item = io.loadmat(mask_path)['mask']   ## R5
      #mask_item = np.rot90(mask_item,k=-1)  ###转

      for i in range(coil):
        mask[i,:,:] = mask_item
 
      print('Sampling rate: ',np.sum(mask_item)/65536)
      #plt.imshow(abs(mask),cmap='gray')
      #plt.show()
      if not os.path.exists(save_path):
        os.mkdir(save_path)  
      write_images(abs(mask_item),osp.join(save_path,os.path.basename(mask_path[:-4])+'mask'+'.png'))
      #========   
        
      
      # ww = io.loadmat('./weight_0.399_T2test.mat')['weight']
      ww = io.loadmat(os.environ.get('SELFMF_WEIGHT_MAT', './data/weight.mat'))['weight']
      weight = np.zeros((coil,256,256))       
      
      for i in range(coil):
        weight[i,:,:] = ww

      Kdata = np.zeros((coil,256,256),dtype=np.complex64)
      Ksample = np.zeros((coil,256,256),dtype=np.complex64)
      zeorfilled_data = np.zeros((coil,256,256),dtype=np.complex64)
      k_w = np.zeros((coil,256,256),dtype=np.complex64)
      for i in range(coil):
        Kdata[i,:,:] = np.fft.fft2(ori_data[i,:,:])# max: 3820.8044
        Ksample[i,:,:] = np.multiply(mask[i,:,:],Kdata[i,:,:]) # max: 3820.8044
        k_w[i,:,:] = k2wgt(Ksample[i,:,:],weight[i,:,:]) # max: 0.42637014            
        zeorfilled_data[i,:,:] = np.fft.ifft2(Ksample[i,:,:])  
    
      zeorfilled_data_sos = np.sqrt(np.sum(np.square(np.abs(zeorfilled_data)),axis=0))
      
      ori_data_sos = ori_data_sos/np.max(np.abs(ori_data_sos))
      zeorfilled_data_sos = zeorfilled_data_sos/np.max(np.abs(zeorfilled_data_sos))  
      
      print('max k_w: ',np.max(np.abs(k_w)))
      #rec_Image = np.clip(rec_Image,-1,1) 
      #write_kdata(Kdata,'kdata')123
      #write_kdata(Ksample,'ksample')
      # psnr_zero:  24.34919633410939 ssim_zero:  0.7610677629148306
      psnr_zero=compare_psnr(255*abs(zeorfilled_data_sos),255*abs(ori_data_sos),data_range=255)
      ssim_zero=compare_ssim(abs(zeorfilled_data_sos),abs(ori_data_sos),data_range=1)
      print('psnr_zero: ',psnr_zero,'ssim_zero: ',ssim_zero)
      write_images(abs(zeorfilled_data_sos),osp.join(save_path,img_name+'Zeorfilled_'+str(round(psnr_zero, 2))+str(round(ssim_zero, 4))+'.png'))
      io.savemat(osp.join(save_path,img_name+'zeorfilled.mat'),{'zeorfilled':zeorfilled_data})
      
      ##=========================================== sake ksample
      MASK = mask
      MASK = np.swapaxes(MASK,0,2)
      MASK = np.swapaxes(MASK,0,1)
      
      Kdata_sake_used = fft2c_noshift(ori) # max: 14.9250
      KtoImg(Kdata_sake_used,osp.join(save_path,img_name+'sake_used_T2img.png'),coil)
      #Kdata_sake_used = Kdata_sake_used/np.max(np.abs(ifft2c(Kdata_sake_used))) + 2.2204e-16
      
      Ksample_sake_used = np.zeros((256,256,coil),dtype=np.complex64)
      for i in range(coil):
        Ksample_sake_used[:,:,i] = np.multiply(MASK[:,:,i],Kdata_sake_used[:,:,i])  
        
        
      ##===========================================
      x_input = np.random.uniform(-1,1,size=(coil,6,256,256))
      
      
      #x_input = sde.prior_sampling(shape).to(device)
      #x_input = np.zeros((coil,6,256,256))
      mask1=io.loadmat(os.environ.get('SELFMF_MASK40_MAT', './data/mask40.mat'))['mask']
      mask1=1-mask1
      mask1[102:151,102:151]=1
      mask2=1-mask1
      mask2[102:151,102:151]=1
      mask3=mask2
      mask3[102:151,102:151]=0
      write_images(mask1,osp.join(save_path,'Mask1'+'.png'))
      
      chan1 = np.zeros((coil,256,256),dtype=np.complex64)
      chan2 = np.zeros((coil,256,256),dtype=np.complex64)
      chan1_w = np.zeros((coil,256,256),dtype=np.complex64)
      chan2_w = np.zeros((coil,256,256),dtype=np.complex64)
      for i in range(coil):
        chan1[i,:,:]=np.multiply(T1_ori_data[i,:,:],1-mask1)+np.multiply(Ksample[i,:,:],mask1) # 
        chan2[i,:,:]=np.multiply(T1_ori_data[i,:,:],1-mask2)+np.multiply(Ksample[i,:,:],mask2)
        chan1_w[i,:,:] = k2wgt(chan1[i,:,:], weight[i,:,:])
        chan2_w[i,:,:] = k2wgt(chan2[i,:,:], weight[i,:,:])


      x_input = np.stack((np.real(chan1_w),np.imag(chan1_w)),1)
      x_input = np.repeat(x_input,3,1)
      
      #堆叠3个通道分别为:T2+T2+T2"
      x_input[:,0,:,:] =  np.real(chan1_w)
      x_input[:,1,:,:] =  np.imag(chan1_w)
      x_input[:,2,:,:] =  np.real(chan2_w)
      x_input[:,3,:,:] =  np.imag(chan2_w)
      x_input[:,4,:,:] =  np.real(k_w)
      x_input[:,5,:,:] =  np.imag(k_w)

 
      x_input = torch.from_numpy(x_input).to(device)
      x_mean = torch.tensor(x_input,dtype=torch.float32).to(device)
      x1 = x_mean
      x2 = x_mean
      x3 = x_mean 
      
          
      max_psnr = float("-inf")
      max_psnr_ssim = float("nan")
      max_ssim = float("-inf")
      
      time1 = time.time()
      for indx in range(sde.N):
      # for indx in range(10):
        print('======== ',indx)
        time2 = time.time()
        t = timesteps[indx]
        vec_t = torch.ones(shape[0], device=t.device) * t
        
        ##======================================================= Predictor
        x, x_mean = predictor_update_fn(x_mean, vec_t, model=model)
        
        x_mean = x_mean.cpu().numpy() # (8,6,256,256)    
        x_mean = np.array(x_mean,dtype=np.float32) 
        
        kw_real = np.zeros((coil,256,256),dtype=np.float32)
        kw_imag = np.zeros((coil,256,256),dtype=np.float32)   
        for i in range(coil):    
          kw_real[i,:,:] = (np.multiply(x_mean[i,0,:,:],mask1)+np.multiply(x_mean[i,2,:,:],mask3)+x_mean[i,4,:,:])/2
          kw_imag[i,:,:] = (np.multiply(x_mean[i,1,:,:],mask1)+np.multiply(x_mean[i,3,:,:],mask3)+x_mean[i,5,:,:])/2
          k_w[i,:,:] = kw_real[i,:,:]+1j*kw_imag[i,:,:]
        
        # DC 1
        k_complex = np.zeros((coil,256,256),dtype=np.complex64)
        k_complex2 = np.zeros((coil,256,256),dtype=np.complex64)        
        for i in range(coil):       
          k_complex[i,:,:] = wgt2k(k_w[i,:,:],weight[i,:,:],Ksample[i,:,:])
          k_complex2[i,:,:] = enforce_acquired_samples(k_complex[i,:,:], Ksample[i,:,:], mask[i,:,:])
        
        # BACK 1
        x_input = np.zeros((coil,6,256,256),dtype=np.float32)
        for i in range(coil): 
          k_w[i,:,:] = k2wgt(k_complex2[i,:,:],weight[i,:,:])
          x_input[i,0,:,:] = np.real(k_w[i,:,:])
          x_input[i,1,:,:] = np.imag(k_w[i,:,:])
          x_input[i,2,:,:] = np.real(k_w[i,:,:])
          x_input[i,3,:,:] = np.imag(k_w[i,:,:])
          x_input[i,4,:,:] = np.real(k_w[i,:,:])
          x_input[i,5,:,:] = np.imag(k_w[i,:,:])
        x_mean = torch.tensor(x_input,dtype=torch.float32).to(device)
      
        ##======================================================= Corrector
        x1,x2,x3,x_mean = corrector_update_fn(x1,x2,x3,x_mean, vec_t, model=model)       
        x_mean = x_mean.cpu().numpy() 
        x_mean = np.array(x_mean,dtype=np.float32)
            
        kw_real = np.zeros((coil,256,256),dtype=np.float32)
        kw_imag = np.zeros((coil,256,256),dtype=np.float32)   
        for i in range(coil):    
          kw_real[i,:,:] = (np.multiply(x_mean[i,0,:,:],mask1)+np.multiply(x_mean[i,2,:,:],mask3)+x_mean[i,4,:,:])/2
          kw_imag[i,:,:] = (np.multiply(x_mean[i,1,:,:],mask1)+np.multiply(x_mean[i,3,:,:],mask3)+x_mean[i,5,:,:])/2
          k_w[i,:,:] = kw_real[i,:,:]+1j*kw_imag[i,:,:]
        
        # DC 2
        k_complex = np.zeros((coil,256,256),dtype=np.complex64)
        k_complex2 = np.zeros((coil,256,256),dtype=np.complex64)
        NCSNpp_Image = np.zeros((coil,256,256),dtype=np.complex64)
        for i in range(coil):       
          k_complex[i,:,:] = wgt2k(k_w[i,:,:],weight[i,:,:],Ksample[i,:,:])
          k_complex2[i,:,:] = enforce_acquired_samples(k_complex[i,:,:], Ksample[i,:,:], mask[i,:,:])
          NCSNpp_Image[i,:,:] = np.fft.ifft2(k_complex2[i,:,:])
        
        #======================================================= SAKE
        # sake intputmask
        Input_Imag = np.copy(NCSNpp_Image)
        Input_Imag = np.swapaxes(Input_Imag,0,2)
        Input_Imag = np.swapaxes(Input_Imag,0,1) 

        size = (Input_Imag).shape
        fctr = size[0]*size[1]
        k_complex3 = np.zeros((size),dtype=np.complex64)
        for i in range(size[2]):
          k_complex3[:,:,i] = (1/np.sqrt(fctr))*np.fft.fft2(Input_Imag[:,:,i])
    
        ksize = [8,8]
        wnthresh = 1.8   #1.8 and 1.5 
        sakeIter = 1
        # sake function
        rec_Image, Krec = sake(k_complex3,Ksample_sake_used, MASK, ksize, wnthresh, sakeIter)

        size = (Input_Imag).shape
        fctr = size[0]*size[1]
        K_end = np.zeros((256,256,coil),dtype=np.complex64)
        for i in range(size[2]):
          K_end[:,:,i] = np.sqrt(fctr)*Krec[:,:,i]
        
        # x_w_kdata = np.zeros((coil,256,256),dtype=np.complex64)
        # for i in range(coil):
        #     x_w_kdata[i, :, :] = Krec.transpose(2,0,1)[i, :, :]

        ## dc
        x_w_dc_kdata = np.zeros((coil,256,256),dtype=np.complex64)
        for i in range(coil):
          x_w_dc_kdata[i, :, :] = enforce_acquired_samples(K_end[:, :, i], Ksample[i, :, :], mask[i, :, :])
        
        x_w_data = np.zeros((coil,256,256),dtype=np.complex64)
        for i in range(coil):
          x_w_data[i, :, :] = np.fft.ifft2(x_w_dc_kdata[i, :, :])
        

        ###  提亮重建后的图
        x_w_data[x_w_data<0.1]+=0.012
        x_w_data[(x_w_data>=0.1)&(x_w_data<=0.3)]-=0.012
        #=======================================================  
        rec_Image_sos = np.sqrt(np.sum(np.square(np.abs(x_w_data)),axis=0))
        # rec_Image_sos = np.sqrt(np.sum(np.square(np.abs(rec_Image)),axis=2))
        rec_Image_sos = rec_Image_sos/np.max(np.abs(rec_Image_sos))
        
        time3 = time.time()
        print("Each iter: %.2f s"%(time3-time2))
        
        # Print PSNR
        psnr = compare_psnr(255*abs(rec_Image_sos),255*abs(ori_data_sos),data_range=255)
        ssim = compare_ssim(abs(rec_Image_sos),abs(ori_data_sos),data_range=1)
        
        print(' PSNR:', psnr,' SSIM:', ssim)  
        write_Data2(psnr,ssim,img_name,save_path) 
        #io.savemat(osp.join(save_path,'rec_Image_sos'+str(psnr)+'.mat'),{'rec_Image_sos':rec_Image_sos})
        
        # Save Result
        if max_ssim<=ssim:
          max_ssim = ssim
        if max_psnr<=psnr:
          max_psnr = psnr
          max_psnr_ssim = ssim
          write_Data('checkpoint',max_psnr,ssim,img_name,save_path) 
          write_images(abs(rec_Image_sos),osp.join(save_path,img_name+'Rec'+'.png'))
          io.savemat(osp.join(save_path,img_name+'kncsn.mat'),{'kncsn':rec_Image})    
        #krec = np.fft.fft2(rec_Image)
        #write_kdata(krec,'k_Rec')

        # BACK 2

        
        x_input = np.zeros((coil,6,256,256),dtype=np.float32)
        for i in range(coil): 
          k_w[i,:,:] = k2wgt(x_w_dc_kdata[i,:,:],weight[i,:,:])
          x_input[i,0,:,:] = np.real(k_w[i,:,:])
          x_input[i,1,:,:] = np.imag(k_w[i,:,:])
          x_input[i,2,:,:] = np.real(k_w[i,:,:])
          x_input[i,3,:,:] = np.imag(k_w[i,:,:])
          x_input[i,4,:,:] = np.real(k_w[i,:,:])
          x_input[i,5,:,:] = np.imag(k_w[i,:,:])
        x_mean = torch.tensor(x_input,dtype=torch.float32).to(device)
        x_mean = x_mean.to(device)          
        
        
        
      time4 = time.time()
      print("All time: %.2f s"%(time4-time1))  
      return x_mean,{"psnr":max_psnr,"ssim":max_psnr_ssim,"zf_psnr":psnr_zero,"zf_ssim":ssim_zero}#inverse_scaler(x_mean if denoise else x), sde.N * (n_steps + 1)

  return pc_sampler



