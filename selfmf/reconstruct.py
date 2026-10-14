

from dataclasses import dataclass, field
import matplotlib.pyplot as plt
import io
import csv
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib
import importlib
import os
import functools
import itertools
import torch
from losses import get_optimizer
from models.ema import ExponentialMovingAverage

import torch.nn as nn
import numpy as np
import tensorflow as tf
import tensorflow_datasets as tfds
import tensorflow_gan as tfgan
import tqdm
import io
import likelihood
import controllable_generation
from utils import restore_checkpoint
sns.set(font_scale=2)
sns.set(style="whitegrid")
from scipy.io import loadmat
import models
from models import utils as mutils
from models import ncsnv2
from models import ncsnpp
from models import ddpm as ddpm_model
from models import layerspp
from models import layers
from models import normalization
from likelihood import get_likelihood_fn
from sde_lib import VESDE, VPSDE, subVPSDE
'''
sampling2_parallel_svd
sampling2_parallel_svd_仅加T1
sampling2_parallel_svd_仅加T1_DCT1
sampling2_parallel_svd_仅加T1_迭代T1
sampling2_parallel_svd_加PD_DCPD
sampling2_parallel_svd_拼接
sampling2_parallel_svd_仅加T1_gy

sampling2_parallel_svd_testT2_gy

'''
import sampling_conditional as sampling_svd
'''
from sampling2_parallel_svd import (ReverseDiffusionPredictor, 
                      LangevinCorrector, 
                      EulerMaruyamaPredictor, 
                      AncestralSamplingPredictor, 
                      NoneCorrector, 
                      NonePredictor,
                      AnnealedLangevinDynamics)
'''
import datasets_wd as datasets
import os.path as osp

sde = 'VESDE' #@param ['VESDE', 'VPSDE', 'subVPSDE'] {"type": "string"}
if sde.lower() == 'vesde':
  from configs.ve import SIAT_kdata_ncsnpp_test as configs  # 修改config
  model_num = 'checkpoint.pth'
  ckpt_filename = os.environ.get('SELFMF_CHECKPOINT', './checkpoints/model.pth')
  config = configs.get_config()  
  sde = VESDE(sigma_min=config.model.sigma_min, sigma_max=config.model.sigma_max, N=config.model.num_scales) ###################################  sde
  sampling_eps = 1e-5


batch_size = 12 #@param {"type":"integer"}
config.training.batch_size = batch_size
config.eval.batch_size = batch_size

random_seed = 0 #@param {"type": "integer"}
 
sigmas = mutils.get_sigmas(config)
scaler = datasets.get_data_scaler(config)
inverse_scaler = datasets.get_data_inverse_scaler(config)
score_model = mutils.create_model(config)

optimizer = get_optimizer(config, score_model.parameters())
ema = ExponentialMovingAverage(score_model.parameters(),
                               decay=config.model.ema_rate)
state = dict(step=0, optimizer=optimizer,
             model=score_model, ema=ema)

state = restore_checkpoint(ckpt_filename, state, config.device)
ema.copy_to(score_model.parameters())

img_size = config.data.image_size
channels = config.data.num_channels
shape = (batch_size, channels, img_size, img_size)
predictor = sampling_svd.ReverseDiffusionPredictor
corrector = sampling_svd.LangevinCorrector
snr = 0.075#0.16 #@param {"type": "number"}
n_steps =  1#@param {"type": "integer"}
probability_flow = False #@param {"type": "boolean"}
sampling_fn = sampling_svd.get_pc_sampler(sde, shape, predictor, corrector,
                                      inverse_scaler, snr, n_steps=n_steps,
                                      probability_flow=probability_flow,
                                      continuous=config.training.continuous,
                                      eps=sampling_eps, device=config.device)




import logging
from collections import OrderedDict
import time
save_path = os.environ.get('SELFMF_OUTPUT_DIR', './outputs')
logger = logging.getLogger("base")
img_path = os.environ.get('SELFMF_TEST_MAT', './data/test.mat')

test_data=loadmat(img_path)
test_results = OrderedDict()                                                                                                                                                                                                                                                                                                                                                     
test_results["psnr"] = []
test_results["ssim"] = []  
test_results["psnr_y"] = []
test_results["ssim_y"] = []

test_results["psnr_zf"] = []
test_results["ssim_zf"] = []
test_times = []

img_name = os.path.splitext(os.path.basename(img_path))[0]
tic = time.time()
x, n = sampling_fn(score_model, test_data,img_name,save_path) #return x_mean,{"psnr":max_psnr,"ssim":max_psnr_ssim,"zf_psnr":psnr_zero,"zf_ssim":ssim_zero}
toc = time.time()
test_time = toc - tic
test_times.append(test_time)
max_psnr = n["psnr"]
max_psnr_ssim = n["ssim"]
psnr_zf = n["zf_psnr"]
ssim_zf = n["zf_ssim"]

test_results["psnr"].append(max_psnr)
test_results["ssim"].append(max_psnr_ssim)
test_results["psnr_zf"].append(psnr_zf)
test_results["ssim_zf"].append(ssim_zf)

logger.info(
      "img:{:15s} - PSNR: {:.2f} dB; SSIM: {:.4f}  *****  零填充: PSNR: {:.2f} dB; SSIM: {:.4f} ***** time: {:.4f} s".format(
          img_name, max_psnr, max_psnr_ssim, psnr_zf, ssim_zf, test_time
      )
  )

ave_psnr = sum(test_results["psnr"]) / len(test_results["psnr"])
ave_ssim = sum(test_results["ssim"]) / len(test_results["ssim"])
ave_psnr_zf = sum(test_results["psnr_zf"]) / len(test_results["psnr_zf"])
ave_ssim_zf = sum(test_results["ssim_zf"]) / len(test_results["ssim_zf"])
ave_time = np.mean(test_times)
logger.info(
    "----Average PSNR/SSIM results----\n\tPSNR: {:.2f} dB; SSIM: {:.4f}*****  零填充: PSNR: {:.2f} dB; SSIM: {:.4f} ***** Average_time: {:.4f}s\n".format(
        ave_psnr, ave_ssim, ave_psnr_zf, ave_ssim_zf, ave_time
    )
)


