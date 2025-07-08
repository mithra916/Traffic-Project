'''import torch
import torch.nn as nn
import torch.optim as optim

class ActorCritic(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(3, 64), nn.ReLU(),
            nn.Linear(64, 32), nn.ReLU(),
        )
        self.actor = nn.Linear(32, 1)
        self.critic = nn.Linear(32, 1)

    def forward(self, x):
        x = self.fc(x)
        return torch.sigmoid(self.actor(x)) * 80, self.critic(x)

model = ActorCritic()
optimizer = optim.Adam(model.parameters(), lr=0.001)

def optimize_with_rl(main_count, incoming_count, base_time):
    input_tensor = torch.tensor([main_count, incoming_count, base_time], dtype=torch.float32)
    action, _ = model(input_tensor)
    return int(action.item())
'''   

# reinforcement_agent.py

import torch
import torch.nn as nn
import torch.optim as optim

class ActorCritic(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(ActorCritic, self).__init__()
        self.actor = nn.Sequential(
            nn.Linear(state_dim, 64),
            nn.ReLU(),
            nn.Linear(64, action_dim),
            nn.Softmax(dim=-1)
        )
        self.critic = nn.Sequential(
            nn.Linear(state_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 1)
        )
        self.optimizer = optim.Adam(self.parameters(), lr=0.001)

    def select_action(self, state):
        state = torch.FloatTensor(state)
        probs = self.actor(state)
        dist = torch.distributions.Categorical(probs)
        action = dist.sample()
        return action.item()

    def update(self, state, action, reward, next_state, done):
        state = torch.FloatTensor(state)
        next_state = torch.FloatTensor(next_state)
        reward = torch.tensor([reward], dtype=torch.float32)
        action = torch.tensor([action])

        value = self.critic(state)
        next_value = self.critic(next_state)
        target = reward + (0.99 * next_value * (1 - int(done)))
        delta = target - value

        critic_loss = delta.pow(2)
        probs = self.actor(state)
        dist = torch.distributions.Categorical(probs)
        actor_loss = -dist.log_prob(action) * delta.detach()

        loss = (actor_loss + critic_loss).mean()
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
