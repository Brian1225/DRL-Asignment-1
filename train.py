# Remember to adjust your student ID in meta.xml
import numpy as np
import pickle
import random
import gym
from simple_custom_taxi_env import SimpleTaxiEnv
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
import random
from utils import DQNAgent, ReplayBuffer


# TODO: Train your own agent
# HINT: If you're using a Q-table, consider designing a custom key based on `obs` to store useful information.
# NOTE: Keep in mind that your Q-table may not cover all possible states in the testing environment.
#       To prevent crashes, implement a fallback strategy for missing keys. 
#       Otherwise, even if your agent performs well in training, it may fail during testing.

class DQNAgentTrainer:
    def __init__(self, state_size, action_size, gamma=0.99, alpha=0.05, eps_start=0.99, eps_end=0.1, decay_rate=0.9995, tau=0.3, buffer_size=10000, batch_size=128, n_episode=10000, update_step=100):
        self.state_size = state_size
        self.action_size = action_size
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.gamma = gamma
        self.alpha = alpha
        self.epsilon = eps_start
        self.eps_end = eps_end
        self.decay_rate = decay_rate
        self.tau = tau
        self.agent = DQNAgent(state_size, action_size)
        self.optimizer = optim.Adam(self.agent.Q.parameters(), lr=alpha)
        self.criterion = nn.MSELoss()
        self.memory = ReplayBuffer(buffer_size, self.device)
        self.batch_size = batch_size
        self.n_episodes = n_episode
        self.update_step = update_step
        
    
    def update(self, state, action, target):
        self.agent.Q.train()
        self.optimizer.zero_grad()
        value = self.agent.Q(state).gather(1, action.unsqueeze(1)).squeeze(1)
        loss = self.criterion(value, target)
        loss.backward()
        self.optimizer.step()

    def train(self):
        self.agent.Q.to(self.device)
        self.agent.target_Q.to(self.device)
        reward_per_episode = [] 
        env = SimpleTaxiEnv()
        for episode in tqdm(range(self.n_episodes)):
            obs, _ = env.reset()
            state = torch.tensor(obs, dtype=torch.float32)
            done = False
            episode_step = 0
            total_reward = 0
            while not done:
                action = self.agent.select_action(state.to(self.device), epsilon=self.epsilon)
                next_obs, reward, done, _ = env.step(action)
                next_state = torch.tensor(next_obs, dtype=torch.float32)
                self.memory.push(state, action, reward, next_state, done)
                state = next_state
                total_reward += reward

                if len(self.memory) > self.batch_size:
                    s, a, r, ns, d = self.memory.sample(self.batch_size)
                    with torch.no_grad():
                        next_q_values = self.agent.target_Q(ns).max(1)[0]
                        target = r + (1 - d.float()) * self.gamma * next_q_values
                    self.update(s, a, target)
                        
                if (episode_step + 1) % self.update_step == 0:
                    for target_param, current_param in zip(self.agent.target_Q.parameters(), self.agent.Q.parameters()):
                        target_param.data.copy_(self.tau * current_param.data + (1 - self.tau) * target_param.data)
                
                episode_step += 1

            self.epsilon = max(self.eps_end, self.decay_rate * self.epsilon)
            reward_per_episode.append(total_reward)
            if (episode + 1) % 100 == 0:
                avg_reward = np.mean(reward_per_episode[-100:])
                print(f"Episode: {episode + 1}, Average reward: {avg_reward}, Epsilon: {self.epsilon}")

        torch.save(self.agent.Q.state_dict(), 'model.pth')

def main():
    trainer = DQNAgentTrainer(16, 6, n_episode=10000)
    trainer.train()

if __name__ == '__main__':
    main()



