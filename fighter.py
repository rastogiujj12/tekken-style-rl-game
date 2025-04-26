import pygame
import numpy as np
import random
from collections import deque
import torch
import torch.nn as nn
import torch.optim as optim

class DQN(nn.Module):
    def __init__(self, input_dim, output_dim):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, output_dim)
        )

    def forward(self, x):
        return self.layers(x)

class Fighter:
    def __init__(self, player, x, y, flip, data, sprite_sheet, animation_steps, attack_sound, screen_width):
        # basic attributes
        self.player        = player
        self.size, self.image_scale, self.offset = data
        self.flip          = flip
        self.animation_list = self.load_images(sprite_sheet, animation_steps)
        self.action        = 0
        self.frame_index   = 0
        self.image         = self.animation_list[0][0]
        self.update_time   = pygame.time.get_ticks()
        self.rect          = pygame.Rect(x, y, 80, 180)
        self.spawn_pos     = (x, y)
        self.vel_y         = 0
        self.jump          = False
        self.attacking     = False
        self.attack_cooldown = 0
        self.hit           = False
        self.health        = 100
        self.alive         = True
        self.death_played  = False
        self.attack_sound  = attack_sound
        self.screen_width  = screen_width

        # DQN setup
        self.device            = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.state_dim         = 4
        self.action_space      = 5
        self.gamma             = 0.99
        self.batch_size        = 64
        self.lr                = 1e-4
        self.epsilon           = 1.0
        self.epsilon_min       = 0.1
        self.epsilon_decay     = 0.995
        self.memory            = deque(maxlen=10000)
        self.train_start       = 1000
        self.update_target_steps = 1000
        self.step_count        = 0

        self.policy_net = DQN(self.state_dim, self.action_space).to(self.device)
        self.target_net = DQN(self.state_dim, self.action_space).to(self.device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()
        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=self.lr)

    def load_images(self, sheet, steps):
        animation_list = []
        for y, count in enumerate(steps):
            frames = []
            for x in range(count):
                img = sheet.subsurface(x * self.size, y * self.size, self.size, self.size)
                frames.append(
                    pygame.transform.scale(img, (self.size * self.image_scale, self.size * self.image_scale))
                )
            animation_list.append(frames)
        return animation_list

    def get_state(self, other):
        dx = (self.rect.x - other.rect.x) / self.screen_width
        dy = (self.rect.y - other.rect.y) / (self.screen_width / 2)
        h1 = self.health / 100.0
        h2 = other.health / 100.0
        return np.array([dx, dy, h1, h2], dtype=np.float32)

    def select_action(self, state):
        if random.random() < self.epsilon:
            return random.randrange(self.action_space)
        state_v = torch.tensor(state, device=self.device).unsqueeze(0)
        with torch.no_grad():
            qvals = self.policy_net(state_v)
        return int(qvals.argmax().cpu().numpy())

    def remember(self, s, a, r, s2, done):
        self.memory.append((s, a, r, s2, done))

    def optimize(self):
        if len(self.memory) < max(self.train_start, self.batch_size):
            return

        batch = random.sample(self.memory, self.batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)

        states_arr      = np.array(states, dtype=np.float32)
        next_states_arr = np.array(next_states, dtype=np.float32)

        states_v      = torch.from_numpy(states_arr).to(self.device)
        next_states_v = torch.from_numpy(next_states_arr).to(self.device)
        actions_v     = torch.tensor(actions, dtype=torch.long, device=self.device).unsqueeze(1)
        rewards_v     = torch.tensor(rewards, dtype=torch.float32, device=self.device).unsqueeze(1)
        dones_v       = torch.tensor(dones, dtype=torch.float32, device=self.device).unsqueeze(1)

        q_pred = self.policy_net(states_v).gather(1, actions_v)
        with torch.no_grad():
            q_next   = self.target_net(next_states_v).max(1)[0].unsqueeze(1)
            q_target = rewards_v + (1.0 - dones_v) * self.gamma * q_next

        loss = nn.MSELoss()(q_pred, q_target)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

    def move(self, other, round_over):
        if round_over:
            return

        # auto face opponent
        self.flip = other.rect.x < self.rect.x

        state  = self.get_state(other)
        action = self.select_action(state)
        reward = 0.0
        done   = False

        SPEED, GRAVITY = 10, 2
        dx, dy = 0, 0
        self.vel_y += GRAVITY
        dy       += self.vel_y

        if action == 0:
            dx = -SPEED
        elif action == 1:
            dx = SPEED
        elif action == 2 and not self.jump:
            self.vel_y = -30
            self.jump  = True
        elif action in (3, 4) and self.attack_cooldown == 0:
            self.attacking = True
            self.attack_sound.play()
            rect = (pygame.Rect(self.rect.right, self.rect.y, 3*self.rect.width, self.rect.height)
                    if not self.flip else
                    pygame.Rect(self.rect.x - 3*self.rect.width, self.rect.y, 3*self.rect.width, self.rect.height))
            if rect.colliderect(other.rect):
                other.health -= 10
                reward = 1.0
            else:
                reward = -0.1
            self.attack_cooldown = 20

        # block overlap
        new_x = self.rect.x + dx
        temp = self.rect.copy()
        temp.x = new_x
        if temp.colliderect(other.rect): dx = 0

        self.rect.x = max(0, min(self.rect.x + dx, self.screen_width - self.rect.width))
        self.rect.y = max(0, min(self.rect.y + dy, (self.screen_width/2) - self.rect.height))
        if self.rect.y >= (self.screen_width/2) - self.rect.height:
            self.jump = False

        if self.health <= 0:
            done = True
            self.alive = False

        next_state = self.get_state(other)
        self.remember(state, action, reward, next_state, done)
        self.optimize()

        self.step_count += 1
        if self.step_count % self.update_target_steps == 0:
            self.target_net.load_state_dict(self.policy_net.state_dict())

        self.attack_cooldown = max(0, self.attack_cooldown - 1)

    def update(self):
        # death animation: play once and then hold last frame
        if not self.alive:
            if not self.death_played:
                # advance through death frames
                self.update_action(6)
                if pygame.time.get_ticks() - self.update_time > 50:
                    self.frame_index += 1
                    self.update_time = pygame.time.get_ticks()
                    if self.frame_index >= len(self.animation_list[6]):
                        self.frame_index = len(self.animation_list[6]) - 1
                        self.death_played = True
            # hold on last death frame
            self.image = self.animation_list[6][self.frame_index]
            return

        # normal animations
        if self.hit:
            self.update_action(5)
        elif self.attacking:
            self.update_action(3)
        elif self.jump:
            self.update_action(2)
        else:
            self.update_action(1 if self.vel_y == 0 else 0)

        if pygame.time.get_ticks() - self.update_time > 50:
            self.frame_index += 1
            self.update_time = pygame.time.get_ticks()
            if self.frame_index >= len(self.animation_list[self.action]):
                self.frame_index = 0
                if self.action in [3, 4]:
                    self.attacking = False
                if self.action == 5:
                    self.hit = False

        self.image = self.animation_list[self.action][self.frame_index]

    def update_action(self, new_action):
        if new_action != self.action:
            self.action = new_action
            self.frame_index = 0
            self.update_time = pygame.time.get_ticks()

    def draw(self, surf):
        img = pygame.transform.flip(self.image, self.flip, False)
        surf.blit(img, (self.rect.x - self.offset[0] * self.image_scale,
                        self.rect.y - self.offset[1] * self.image_scale))

    def reset(self):
        self.health       = 100
        self.alive        = True
        self.death_played = False
        self.vel_y        = 0
        self.jump         = False
        self.attacking    = False
        self.attack_cooldown = 0
        self.hit          = False
        self.rect.x, self.rect.y = self.spawn_pos
