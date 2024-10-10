import sys
import gym
import torch
import random
import numpy as np
from collections import deque
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.autograd import Variable
from torchvision import transforms
from prioritized_memory import Memory


""" Training Parameters """
EPISODES      = 250
DISCOUNT      = 0.99
LEARNING_RATE = 0.001
MEMORY_SIZE   = 10_000
EPSILON_START = 1.0
EPSILON_END   = 0.1
EPSILON_DECAY_PERIOD = 5_000
BATCH_SIZE    = 64
TRAIN_START   = 1_000
W1 = 0.2
W2 = 0.8
