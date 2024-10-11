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

        """ here we construct the prioritization score to be used for transitions rankings """
        td_error = torch.abs(new_q - current_qs[0][action])
        be_error = torch.abs(new_q - current_qs)
        mean_be_error = torch.mean(be_error)
        weighted_error = self.w1 * td_error + self.w2 * mean_be_error

        self.memory.add(weighted_error, (state, action, reward, next_state, done)) #torch.ones(1,1)

    def train_model(self):
        if self.epsilon > self.epsilon_min:
            self.epsilon -= self.epsilon_decay

        mini_batch, idxs, is_weights = self.memory.sample(self.batch_size)
        
        if mini_batch[-1] == 0 and len(mini_batch) > 1:
            mini_batch[-1] = mini_batch[-2]

        mini_batch = np.array(mini_batch, dtype=object)

        states = np.array([ss[0] for ss in mini_batch]).reshape(self.batch_size, self.state_size)
        actions = np.array([ss[1] for ss in mini_batch])
        rewards = np.array([ss[2] for ss in mini_batch])
        next_states = np.array([ss[3] for ss in mini_batch]).reshape(self.batch_size, self.state_size)
        dones = np.array([ss[4] for ss in mini_batch])

        dones = dones.astype(int)

        states = torch.Tensor(states)
        states = Variable(states).float()
        pred = self.model(states)

        a = torch.LongTensor(actions).view(-1, 1)

        one_hot_action = torch.FloatTensor(self.batch_size, self.action_size).zero_()
        one_hot_action.scatter_(1, a, 1)

        pred = torch.sum(pred.mul(Variable(one_hot_action)), dim=1)

        next_states = torch.Tensor(next_states)
        next_states = Variable(next_states).float()
        next_pred = self.target_model(next_states).data

        rewards = torch.FloatTensor(rewards)
        dones = torch.FloatTensor(dones)

        target = rewards + (1 - dones) * self.discount_factor * next_pred.max(1)[0]
        target = Variable(target)

        errors = torch.abs(pred - target).data.numpy()

        for i in range(self.batch_size):
            idx = idxs[i]
            self.memory.update(idx, errors[i])

        loss = (torch.FloatTensor(is_weights) * F.mse_loss(pred, target)).mean()
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

if __name__ == "__main__":
    
    env = gym.make('CartPole-v0')
    
    state_size = env.observation_space.shape[0]
    action_size = env.action_space.n
    model = DQN(state_size, action_size)
    agent = DQNAgent(state_size, action_size)
    
    scores, episodes = [], []
    ep_rewards = []
    avg_rewards = []
    steps = []

    """
    Initialization of gradient-based weight-adjustment
    """
    agent.w1 = 0.2
    agent.w2 = 0.8
    zeta = 2.2
    lr = 0.001

    for e in range(EPISODES):
        done = False
        step = 0
        ep_reward = 0
        
        state, _ = env.reset(seed=42)
        state = np.reshape(state, [1, state_size])

        """
        Gradient-based optimization of designated weights to the TD and BE errors
        """
        dw1 = 2 * (zeta - (agent.w1 / agent.w2)) * (-1 / (agent.w1 + agent.w2)**2)
        dw2 = -dw1
        agent.w1 -= dw1 * lr
        agent.w2 -= dw2 * lr

        while step < 200:

            if done:
                break
            step += 1

            action = agent.get_action(state)
            next_state, reward, done, _, _ = env.step(action)
            next_state = np.reshape(next_state, [1, state_size])      
                
            ep_reward += reward

            agent.append_sample(state, action, reward, next_state, done)

            if agent.memory.tree.n_entries >= agent.train_start:
                agent.train_model()

            state = next_state

        steps.append(step)

        if e % 10 ==0 and e > 1:
            print(f"Ep: {e}, Avg.: {np.mean(ep_rewards[-10:])}")
            avg_rewards.append(np.mean(ep_rewards[-10:]))

        ep_rewards.append(ep_reward)
        episodes.append(e)            
        agent.update_target_model()
