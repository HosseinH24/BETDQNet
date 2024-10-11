"""
Oct 11, 2024
Pytorch implementation of BETDQNet for the MinAtar experiments
"""

import numpy as np
import torch
import torch.nn as nn
import gymnasium as gym
from copy import deepcopy
from collections import deque
import random
import torch.nn.functional as f
import tensorflow as tf
import datetime
import time

from prioritized_memory import Memory
from torch.autograd import Variable

device = torch.device("mps" if torch.backends.mps.is_available()
							else "cuda" if torch.cuda.is_available()
							else "cpu")
print('Running on "{}"'.format(device))

"""
Calling the environment name and setting the log directory for the tensorflow  logs
"""
env = gym.make('MinAtar/Freeway-v1')
current_time = datetime.datetime.now().strftime(" %Y%m%d-%H%M%S")
log_dir = 'logs/freeway_bet_1' + current_time
summary_writer = tf.summary.create_file_writer(log_dir)

"""
Environment parameters
"""
INPUT_CHANNELS = 7
ACTION_SIZE = env.action_space.n

"""
Training parameters
"""
LEARNING_RATE = 0.00025
MEMORY_SIZE = 100_000
BATCH_SIZE = 32
DISCOUNT = 0.99
EPSILON_START = 1.0
EPSILON_END = 0.1
EPSILON_DECAY_PERIOD = 100_000
TARGET_UPDATE_FREQUENCY = 1_000
TRAIN_FREQUENCY = 1
LEARNING_START = 5_000
TAU = 1.0
NUM_FRAMES = 2_000_000

"""
A CNN model as the Q function approximator
"""
class ConvNet(nn.Module):
    def __init__(self, in_channels, num_actions):

        super(ConvNet, self).__init__()
      
        self.conv = nn.Conv2d(in_channels, 16, kernel_size=3, stride=1)

        def size_linear_unit(size, kernel_size=3, stride=1):
            return (size - (kernel_size - 1) - 1) // stride + 1

        num_linear_units = size_linear_unit(10) * size_linear_unit(10) * 16
        self.fc_hidden = nn.Linear(in_features=num_linear_units, out_features=128)

        self.output = nn.Linear(in_features=128, out_features=num_actions)

    def forward(self, x):
        x = f.relu(self.conv(x))
        x = nn.Flatten()(x)
        x = f.relu(self.fc_hidden(x))

        return self.output(x)

"""
The DQN agent class
We borrowed the prioritized_memory and SumTree from: https://github.com/rlcode/per/tree/master
"""
class DQNAgent():
	def __init__(self):
		self.model = ConvNet(INPUT_CHANNELS, ACTION_SIZE).to(device)
		self.target_model = deepcopy(self.model).to(device)
		self.optimizer = torch.optim.Adam(self.model.parameters(), lr = LEARNING_RATE)
		#self.optimizer = torch.optim.RMSprop(self.model.parameters(), lr=LEARNING_RATE, alpha=0.95, centered=True, eps=0.95)
		#self.memory = deque(maxlen=MEMORY_SIZE)
		self.memory = Memory(MEMORY_SIZE)
		self.epsilon = 1.0
		self.gamma = DISCOUNT
		self.loss_fcn = nn.MSELoss()

		self.w1 = 0.1
		self.w2 = 0.9

	def add(self, transition):
    """
    method to add an entire transition to the buffer 
    """
		self.memory.append(transition)

	def act(self, state):
    """
    the typical epsilon-greedy exploration to pick up actions
    """
		if np.random.random() < self.epsilon:
			action = env.action_space.sample()
		else:
			with torch.no_grad():
				state = torch.FloatTensor(state).unsqueeze(0).to(device)
				action = torch.argmax(self.model(state)).cpu().numpy().item()

		return action

	def append_sample(self, state, action, reward, next_state, done):
    """
    we construct the weighted TD+BE error score and append it along with a transition,
    to rank transitions based on
    """
        
		s = torch.FloatTensor(state).unsqueeze(0).to(device)
		ss = torch.FloatTensor(next_state).unsqueeze(0).to(device)
		current_qs = self.model(s).data
		future_qs = self.target_model(ss).data

		if not done:
			max_future_q = torch.max(future_qs)
			new_q = reward + self.gamma * max_future_q
		else:
			new_q = reward

		td_error = torch.abs(new_q - current_qs[0][action])
		be_error = torch.abs(new_q - current_qs)
	        mean_be_error = torch.mean(be_error)
	        weighted_error = self.w1 * td_error + self.w2 * mean_be_error
		total_error = f.relu(total_error).cpu() # pass the error through the ReLU activation so as to make sure scores are non-negative

		self.memory.add(total_error, (state, action, reward, next_state, done)) #torch.ones(1,1)

	def train(self):
		"""
    self.memory is basically the memory called by the prioritized_memory.py,
    for which the sample method returns priortized samples based on the total_error appended to each transition
    """
		mini_batch, idxs, is_weights = self.memory.sample(BATCH_SIZE)
		

		current_states = np.array([ss[0] for ss in mini_batch])
		actions = np.array([ss[1] for ss in mini_batch])
		rewards = np.array([ss[2] for ss in mini_batch])
		next_states = np.array([ss[3] for ss in mini_batch])
		dones = np.array([ss[4] for ss in mini_batch])

		current_states = torch.FloatTensor(current_states).to(device)
		actions = torch.tensor(actions, dtype=torch.int64).unsqueeze(1).to(device)
		rewards = torch.FloatTensor(rewards).to(device)
		next_states = torch.FloatTensor(next_states).to(device)
		dones = torch.FloatTensor(dones).to(device)

		with torch.no_grad():
			target_Qs = rewards + (1 - dones) * self.gamma * torch.max(self.target_model(next_states), dim=1).values

		current_Qs = self.model(current_states).gather(1, actions)
		target_Qs = target_Qs.reshape(-1,1)

		loss = self.loss_fcn(target_Qs, current_Qs)

		self.optimizer.zero_grad()
		loss.backward()
		self.optimizer.step()

		return loss.item()

	def soft_update(self):
    """
    Update the target network parameters to get closer to the main network
    here TAU sets the level of adjustment for the target's parameters
    """
		for target_param, local_param in zip(self.target_model.parameters(), self.model.parameters()):
			target_param.data.copy_(TAU * local_param.data + (1.0 - TAU) * target_param.data)

	def get_state(self, s):
		return s.reshape(s.shape[2], s.shape[0], s.shape[1])

agent = DQNAgent()

if __name__ == "__main__":
	
	total_step = 0
	episode = 0
	ep_rewards = []

	while total_step < NUM_FRAMES:

		ep_reward = 0

		state, _= env.reset()
		state = agent.get_state(state)
		done = False
		
		while not done:
			action = agent.act(state)
			next_state, reward, done, _, _ = env.step(action)
			next_state = agent.get_state(next_state)
			
			agent.append_sample(state, action, reward, next_state, done)

			total_step += 1
			ep_reward += reward

			state = next_state

			if total_step > LEARNING_START and (total_step % TRAIN_FREQUENCY == 0) and agent.memory.memory_length() > BATCH_SIZE:
				loss = agent.train()
			else:
				loss = 0
			if total_step % TARGET_UPDATE_FREQUENCY == 0:
				agent.soft_update()
			if total_step < EPSILON_DECAY_PERIOD:
				agent.epsilon -= (EPSILON_START - EPSILON_END) / EPSILON_DECAY_PERIOD
			
      """
      adjusting the weights of TD and BE using a gradient-based optimization
      """
      dw1 = 2 * (1.2 - (agent.w1 / agent.w2)) * (-1 / (agent.w1 + agent.w2)**2)
			dw2 = -dw1
			agent.w1 -= dw1 * 0.000001
			agent.w2 -= dw2 * 0.000001

		episode += 1
		ep_rewards.append(ep_reward)
		if episode % 100 == 0:
			end_time = time.time()
			with summary_writer.as_default():
			    tf.summary.scalar('reward', np.mean(ep_rewards[-100:]), step=total_step)
			    tf.summary.scalar('epsilon', agent.epsilon, step=total_step)
			    tf.summary.scalar('loss', loss, step=total_step)

			print("Episode: {}, Frames: {}, Epsilon: {:.2f}, Reward: {:.2f}".format(episode, total_step, agent.epsilon, np.mean(ep_rewards[-100:])))
