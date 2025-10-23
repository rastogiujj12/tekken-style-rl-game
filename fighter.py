import pygame
import numpy as np
import random
from collections import deque
import torch
import torch.nn as nn
import torch.optim as optim
import os

class DQN(nn.Module):
    def __init__(self, input_dim, output_dim):
        super().__init__()
        self.feature = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU()
        )
        self.value_stream = nn.Sequential(
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, 1)
        )
        self.adv_stream = nn.Sequential(
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, output_dim)
        )

    def forward(self, x):
        f = self.feature(x)
        value = self.value_stream(f)
        adv = self.adv_stream(f)
        return value + (adv - adv.mean(dim=1, keepdim=True))

class Fighter:
    def __init__(self, player, x, y, flip, data, sprite_sheet, animation_steps, attack_sound, screen_width, role, training_phase, continue_from_episode = 0):
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
        self.episode_reward = 0.0

        # differentiate roles
        self.role = role
        self.training_phase = training_phase

        # DQN setup
        self.device            = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.state_dim         = 4
        self.action_space      = 5
        self.gamma             = 0.99
        self.batch_size        = 64
        self.lr                = 1e-4
        self.epsilon           = 1.0
        self.epsilon_min       = 0.1
        self.epsilon_decay     = 0.9995
        self.epsilon_linear_end = 0.05
        self.epsilon_anneal_episodes = 800  # reach final eps by N episodes
        self.current_episode = 0
        self.memory            = deque(maxlen=10000)
        self.train_start       = 256
        self.update_target_steps = 200
        self.step_count        = 0

        # if self.training_phase==1 or self.role=="enemy":

        self.policy_net = DQN(self.state_dim, self.action_space).to(self.device)
        self.target_net = DQN(self.state_dim, self.action_space).to(self.device)
        
        npc_model_path = None
        npc_optim_path = None

        # if continue_from_episode>0:
        if self.role=="enemy":
            npc_model_path = f"weights/player_2/phase_2/model/_ep_{continue_from_episode}.pth"
            npc_optim_path = f"weights/player_2/phase_2/optimizer/_ep_{continue_from_episode}.pth"
        else: 
            npc_model_path = f"weights/player_1/phase_1/model/_ep_{continue_from_episode}.pth"
            npc_optim_path = f"weights/player_1/phase_1/optimizer/_ep_{continue_from_episode}.pth"

        print("path", npc_model_path, npc_optim_path)


        if os.path.exists(npc_model_path):
                self.policy_net.load_state_dict(torch.load(npc_model_path, map_location=self.device))
                print("[INFO] Loaded NPC policy weights from Phase 1")
                if os.path.exists(npc_optim_path):
                    self.optimizer = optim.Adam(self.policy_net.parameters(), 
                                                lr=self.lr,
                                                betas=(0.9, 0.999),
                                                eps=1e-08,
                                                weight_decay=0.0
                                            )
                    self.optimizer.load_state_dict(torch.load(npc_optim_path, map_location=self.device))
                    print(f"[INFO] Loaded {self.role} optimizer state from Phase 1")
        else:
            # fallback if optimizer not found
            print(f"[WARN] {self.role} optimizer state not found. Using default optimizer.")
            self.optimizer = optim.Adam(self.policy_net.parameters(), 
                                            lr=self.lr,
                                            betas=(0.9, 0.999),
                                            eps=1e-08,
                                            weight_decay=0.0
                                        )
            self.target_net.load_state_dict(self.policy_net.state_dict())
        
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()

        if self.training_phase==2 and self.role=="player":
            self.optimizer = None


    def anneal_epsilon(self):
        self.epsilon = max(
            self.epsilon_linear_end,
            1 - (self.current.episode / self.epsilon_anneal_episodes) * (1 - self.epsilon_linear_end)
        )
        # if self.current_episode >= self.epsilon_anneal_episodes:
        #     self.epsilon = self.epsilon_linear_end
        # else:
        #     start = 1.0
        #     end = self.epsilon_linear_end
        #     t = self.current_episode / max(1, self.epsilon_anneal_episodes)
        #     self.epsilon = start + t * (end - start)  # linear interpolation

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
        vy = self.vel_y / 30.0
        atk_cd = self.attack_cooldown / 20.0
        return np.array([dx, dy, h1, h2, vy, atk_cd], dtype=np.float32)

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
            return None, None

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

        # loss = nn.MSELoss()(q_pred, q_target)

        # Use Huber loss for stability
        loss_fn = nn.SmoothL1Loss()
        loss = loss_fn(q_pred, q_target)

        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.policy_net.parameters(), max_norm=10.0)
        self.optimizer.step()

        # if self.epsilon > self.epsilon_min:
        #     self.epsilon *= self.epsilon_decay

        avg_q_pred = q_pred.mean().item()
        return float(loss.item()), avg_q_pred
        # return float(loss.item())

    def move(self, other, round_over, elapsed):
        if round_over:
            return

        # auto face opponent
        self.flip = other.rect.x < self.rect.x

        state  = self.get_state(other)
        action = self.select_action(state)
        reward = 0.0
        done   = False
        prev_dx = abs(self.rect.x - other.rect.x)

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
            # self.attack_sound.play()
           
            rect = (pygame.Rect(self.rect.right, self.rect.y, 3*self.rect.width, self.rect.height)
                    if not self.flip else
                    pygame.Rect(self.rect.x - 3*self.rect.width, self.rect.y, 3*self.rect.width, self.rect.height))
            
            # reward
            new_dx = abs(self.rect.x - other.rect.x)
            shaping = (prev_dx - new_dx) * 2  # small positive if you moved closer
            reward = shaping

            if not self.training_phase == 1:
                # reward = balance_reward + alive_bonus + fast_end_penalty
                target_time = 60.0
                sigma = 20.0  # spread in seconds
                duration_reward = 0.2 * np.exp(-((elapsed - target_time)**2) / (2 * sigma**2)) - 0.1
                reward += duration_reward

            if rect.colliderect(other.rect):
                other.health -= 10/3

                if self.training_phase==1 or self.role == "player":
                    reward = 1.0
                else :
                    # Phase 2 (DDA): aim for fun balance
                    health_diff = abs(self.health - other.health) / 100.0

                    # Reward highest when healths are close
                    balance_reward = 1.0 - (health_diff * 2.5)  # penalize big gaps
                    balance_reward = np.clip(balance_reward, -1.0, 1.0)
                    reward += balance_reward
                    # # Add small engagement bonus if both still alive
                    # alive_bonus = 0.1 if self.alive and other.alive else 0.0

                    # # Slight penalty if fight ends too fast (boring)
                    # fast_end_penalty = -0.5 if not other.alive or not self.alive else 0.0


            else:
                reward += -0.3

            reward *= 5
            reward = max(-5.0, min(reward, 5.0))
            self.episode_reward += reward
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
        
        if not (self.training_phase == 2 and self.role == "player"):
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
        self.episode_reward = 0.0
        self.health       = 100
        self.alive        = True
        self.death_played = False
        self.vel_y        = 0
        self.jump         = False
        self.attacking    = False
        self.attack_cooldown = 0
        self.hit          = False
        self.rect.x, self.rect.y = self.spawn_pos
