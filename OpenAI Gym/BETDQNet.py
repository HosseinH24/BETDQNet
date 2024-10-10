"""
Oct 10, 2024
Pytorch implementation of BETDQNet for OpenAI Gym environments. 
BETDQNet uses both Bellman and TD errors to prioritize samples, each of which is weighted dynamically.
Weights are adjusted through a gradient-based optimization mechanisms, to first encourage exploration and then focus on exploitation.
"""

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

""" Feedforward network as the Q function """
class DQN(nn.Module):
    def __init__(self, state_size, action_size):
        super(DQN, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(state_size, 50),
            nn.ReLU(),
            nn.Linear(50, 50),
            nn.ReLU(),
            nn.Linear(50, action_size)
        )

    def forward(self, x):
        return self.fc(x)

"""
We define a class for the agent. 
The agent uses prioritized replay memory borrowed from: https://github.com/rlcode/per/tree/master
to search for the prioritized samples by means of the proposed BETDQNet prioritization score.
"""
class DQNAgent():
    def __init__(self, state_size, action_size):
        
        self.render = False
        self.load_model = False

        """ get the size of state and action spaces """
        self.state_size = state_size
        self.action_size = action_size

        """ training hyperparameters """
        self.discount_factor = DISCOUNT
        self.learning_rate = LEARNING_RATE
        self.memory_size = MEMORY_SIZE
        self.epsilon = EPSILON_START
        self.epsilon_min = EPSILON_END
        self.explore_step = EPSILON_DECAY_PERIOD
        self.epsilon_decay = (self.epsilon - self.epsilon_min) / self.explore_step
        self.batch_size = BATCH_SIZE
        self.train_start = TRAIN_START

        """ W1 is assigned to the TD error and W2 to the BE """
        self.w1 = W1
        self.w2 = W2
        self.td_buffer = deque(maxlen=self.memory_size)
        self.be_buffer = deque(maxlen=self.memory_size)

        self.memory = Memory(self.memory_size)

        self.model = DQN(state_size, action_size)
        self.target_model = DQN(state_size, action_size)
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate)

        self.update_target_model()

    
    def update_target_model(self):
      """ to periodically update the target model """
        self.target_model.load_state_dict(self.model.state_dict())
      
    def get_action(self, state):
      """ the agent follows an epsilon-greedy exploration """
        if np.random.rand() <= self.epsilon:
            return random.randrange(self.action_size)
        else:
            state = torch.from_numpy(state)
            state = Variable(state).float().cpu()
            q_value = self.model(state)
            _, action = torch.max(q_value, 1)
            return int(action)

    def append_sample(self, state, action, reward, next_state, done):
        """ each sample to be appended in the replay memory, will be accompanied by its associated weighted prioritization score
                                                                    based on the TD and BE with designated W1 and W2 weights """
        
        current_qs = self.model(Variable(torch.FloatTensor(state))).data
        future_qs = self.target_model(Variable(torch.FloatTensor(next_state))).data

        if not done:
            max_future_q = torch.max(future_qs)
            new_q = reward + self.discount_factor * max_future_q
        else:
            new_q = reward

        td_error = torch.abs(new_q - current_qs[0][action])
        be_error = torch.abs(torch.mean(current_qs) - torch.mean(future_qs))
        total_error = self.w1 * td_error / 5 + self.w2 * be_error / 5

        ####### ReLo Loss Prioritization ################
        '''q_next = self.target_model(Variable(torch.FloatTensor(next_state))).data
        qq = self.target_model(Variable(torch.FloatTensor(state))).data
        if not done:
            max_future_q = torch.max(q_next)
            q_target = reward + self.discount_factor * max_future_q
        else:
            q_target = reward
        q = self.model(Variable(torch.FloatTensor(state))).data

        td_loss = torch.mean((q - q_target)**2)
        second_loss = torch.mean((qq - q_target)**2)

        total_error = td_loss - second_loss'''
        ################################################

        self.memory.add(total_error, (state, action, reward, next_state, done)) #torch.ones(1,1)

    # pick samples from prioritized replay memory (with batch_size)
    def train_model(self):
        if self.epsilon > self.epsilon_min:
            self.epsilon -= self.epsilon_decay

        mini_batch, idxs, is_weights = self.memory.sample(self.batch_size)
        
        if mini_batch[-1] == 0 and len(mini_batch) > 1:
            mini_batch[-1] = mini_batch[-2]


        mini_batch = np.array(mini_batch, dtype=object)
        #print(mini_batch.shape)

        states = np.array([ss[0] for ss in mini_batch]).reshape(self.batch_size, self.state_size)
        actions = np.array([ss[1] for ss in mini_batch])
        rewards = np.array([ss[2] for ss in mini_batch])
        next_states = np.array([ss[3] for ss in mini_batch]).reshape(self.batch_size, self.state_size)
        dones = np.array([ss[4] for ss in mini_batch])
        
        #mini_batch = np.array(mini_batch,dtype=object).transpose()


        # bool to binary
        dones = dones.astype(int)

        # Q function of current state
        states = torch.Tensor(states)
        states = Variable(states).float()
        pred = self.model(states)

        # one-hot encoding
        a = torch.LongTensor(actions).view(-1, 1)

        one_hot_action = torch.FloatTensor(self.batch_size, self.action_size).zero_()
        one_hot_action.scatter_(1, a, 1)

        pred = torch.sum(pred.mul(Variable(one_hot_action)), dim=1)

        # Q function of next state
        next_states = torch.Tensor(next_states)
        next_states = Variable(next_states).float()
        next_pred = self.target_model(next_states).data

        rewards = torch.FloatTensor(rewards)
        dones = torch.FloatTensor(dones)

        # Q Learning: get maximum Q value at s' from target model
        target = rewards + (1 - dones) * self.discount_factor * next_pred.max(1)[0]
        target = Variable(target)

        errors = torch.abs(pred - target).data.numpy()

        # update priority
        for i in range(self.batch_size):
            idx = idxs[i]
            self.memory.update(idx, errors[i])

        self.optimizer.zero_grad()

        # MSE Loss function
        loss = (torch.FloatTensor(is_weights) * F.mse_loss(pred, target)).mean()
        loss.backward()

        # and train
        self.optimizer.step()





















