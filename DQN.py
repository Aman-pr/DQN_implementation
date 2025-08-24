import numpy as np
import random
import torch
import torch.nn as nn
import torch.optim as optim
from collections import deque
import argparse

# -----------------------------
# Environment: Simplified Flappy Bird
# -----------------------------
class FlappyBirdEnv:
    def __init__(self):
        self.reset()

    def reset(self):
        self.bird_y = 0.5  # bird position (normalized height)
        self.bird_v = 0.0  # velocity
        self.pipe_x = 1.0  # pipe horizontal pos (normalized)
        self.pipe_gap_y = np.random.uniform(0.3, 0.7)
        self.t = 0
        return self._get_state()

    def _get_state(self):
        return np.array([self.bird_y, self.bird_v, self.pipe_x, self.pipe_gap_y], dtype=np.float32)

    def step(self, action):
        # action: 0 = do nothing, 1 = flap
        if action == 1:
            self.bird_v = -0.05
        else:
            self.bird_v += 0.003  # gravity

        self.bird_y += self.bird_v
        self.pipe_x -= 0.02
        reward = 0.1  # survival reward
        done = False
        passed_pipe = False

        # check pipe passed
        if self.pipe_x < 0:
            self.pipe_x = 1.0
            self.pipe_gap_y = np.random.uniform(0.3, 0.7)
            passed_pipe = True
            reward += 5.0  # big reward for success

        # check collision (ground, ceiling, or hitting pipe)
        if self.bird_y <= 0 or self.bird_y >= 1:
            done = True
        elif abs(self.pipe_x - 0.5) < 0.05:  # near pipe
            if abs(self.bird_y - self.pipe_gap_y) > 0.2:  # outside gap
                done = True

        if done:
            reward -= 5.0  # death penalty

        return self._get_state(), reward, done, {}

# -----------------------------
# DQN Agent
# -----------------------------
class DQN(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(DQN, self).__init__()
        self.fc1 = nn.Linear(state_dim, 128)
        self.fc2 = nn.Linear(128, 128)
        self.fc3 = nn.Linear(128, action_dim)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        return self.fc3(x)

class Agent:
    def __init__(self, state_dim, action_dim, gamma=0.99, lr=1e-3,
                 eps_start=1.0, eps_end=0.05, eps_decay=0.995):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.gamma = gamma
        self.epsilon = eps_start
        self.eps_min = eps_end
        self.eps_decay = eps_decay

        self.memory = deque(maxlen=50000)
        self.batch_size = 64

        self.policy_net = DQN(state_dim, action_dim)
        self.target_net = DQN(state_dim, action_dim)
        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=lr)
        self.loss_fn = nn.MSELoss()

        self.update_target()

    def update_target(self):
        self.target_net.load_state_dict(self.policy_net.state_dict())

    def act(self, state):
        if random.random() < self.epsilon:
            return random.randrange(self.action_dim)
        state = torch.tensor(state, dtype=torch.float32).unsqueeze(0)
        with torch.no_grad():
            q_values = self.policy_net(state)
        return q_values.argmax().item()

    def remember(self, s, a, r, s2, done):
        self.memory.append((s, a, r, s2, done))

    def replay(self):
        if len(self.memory) < self.batch_size:
            return
        batch = random.sample(self.memory, self.batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)

        states = torch.tensor(states, dtype=torch.float32)
        actions = torch.tensor(actions, dtype=torch.int64).unsqueeze(1)
        rewards = torch.tensor(rewards, dtype=torch.float32).unsqueeze(1)
        next_states = torch.tensor(next_states, dtype=torch.float32)
        dones = torch.tensor(dones, dtype=torch.float32).unsqueeze(1)

        q_values = self.policy_net(states).gather(1, actions)
        next_q = self.target_net(next_states).max(1)[0].detach().unsqueeze(1)
        target = rewards + (1 - dones) * self.gamma * next_q

        loss = self.loss_fn(q_values, target)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        # decay epsilon slowly
        if self.epsilon > self.eps_min:
            self.epsilon *= self.eps_decay

# -----------------------------
# Training loop
# -----------------------------
def train_agent(episodes=2000, target_update=20, model_file="flappy_dqn.pt"):
    env = FlappyBirdEnv()
    agent = Agent(state_dim=4, action_dim=2)

    scores = []
    for ep in range(episodes):
        state = env.reset()
        total_reward = 0
        score = 0
        for t in range(1000):  # limit steps
            action = agent.act(state)
            next_state, reward, done, _ = env.step(action)
            agent.remember(state, action, reward, next_state, done)
            agent.replay()
            state = next_state
            total_reward += reward
            if reward > 1:  # passed pipe
                score += 1
            if done:
                break

        scores.append(score)
        if ep % target_update == 0:
            agent.update_target()

        avg_score = np.mean(scores[-20:])
        print(f"Ep {ep}, Score {score}, Reward {total_reward:.2f}, Avg20 {avg_score:.2f}, Eps {agent.epsilon:.3f}")

    torch.save(agent.policy_net.state_dict(), model_file)
    print(f"Model saved to {model_file}")

# -----------------------------
# Evaluation loop
# -----------------------------
def evaluate(model_file="flappy_dqn.pt", episodes=20):
    env = FlappyBirdEnv()
    agent = Agent(state_dim=4, action_dim=2)
    agent.policy_net.load_state_dict(torch.load(model_file))
    agent.epsilon = 0.0  # greedy play

    scores = []
    for ep in range(episodes):
        state = env.reset()
        score = 0
        for t in range(1000):
            action = agent.act(state)
            next_state, reward, done, _ = env.step(action)
            state = next_state
            if reward > 1:
                score += 1
            if done:
                break
        scores.append(score)
        print(f"Eval Ep {ep}, Score {score}")
    print("Avg score:", np.mean(scores))

# -----------------------------
# Main
# -----------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--episodes", type=int, default=2000)
    parser.add_argument("--eval", action="store_true")
    parser.add_argument("--model", type=str, default="flappy_dqn.pt")
    args = parser.parse_args()

    if args.eval:
        evaluate(args.model)
    else:
        train_agent(episodes=args.episodes, model_file=args.model)
