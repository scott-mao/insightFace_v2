import os

import torch

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')  # sets device for model and PyTorch tensors

# Model parameters
image_w = 112
image_h = 112
channel = 3
emb_size = 512

# Training parameters
num_workers = 1  # for data-loading; right now, only 1 works with h5py
grad_clip = 5.  # clip gradients at an absolute value of
print_freq = 100  # print training/validation stats  every __ batches
checkpoint = None  # path to checkpoint, None if none

# Data parameters
num_classes = 93431
num_samples = 5179510
DATA_DIR = '/home/tupm/Documents'
# faces_ms1m_folder = 'data/faces_ms1m_112x112'
faces_ms1m_folder = '/home/tupm/HDD/datasets/2d_face/insight/face/faces_emore'
path_imgidx = os.path.join(faces_ms1m_folder, 'train.idx')
path_imgrec = os.path.join(faces_ms1m_folder, 'train.rec')
IMG_DIR = '/home/tupm/Documents/augmented'
pickle_file = '/home/tupm/HDD/datasets/2d_face/insight/face/faces_emore/data/faces_ms1m_112x112.pickle'
