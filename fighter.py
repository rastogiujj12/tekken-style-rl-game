import pygame
import numpy as np
import random
from collections import deque
import torch
import torch.nn as nn
import torch.optim as optim
import os

class RunningMeanStd:
    # simple running mean/std for input normalization
    def __init__(self, eps=1e-4, shape=()):
        self.mean = np.zeros(shape, 'float64')
        self.var = np.ones(shape, 'float64')
        self.count = eps

    def update(self, x: np.ndarray):
        batch_mean = np.mean(x, axis=0)
        batch_var = np.var(x, axis=0)
        batch_count = x.shape[0]
        self._update_from_moments(batch_mean, batch_var, batch_count)

    def _update_from_moments(self, batch_mean, batch_var, batch_count):
        delta = batch_mean - self.mean
        tot_count = self.count + batch_count

        new_mean = self.mean + delta * batch_count / tot_count
        m_a = self.var * (self.count)
        m_b = batch_var * (batch_count)
        M2 = m_a + m_b + np.square(delta) * self.count * batch_count / tot_count
        new_var = M2 / (tot_count)

        self.mean = new_mean
        self.var = new_var
        self.count = tot_count

    def normalize(self, x):
        return (x - self.mean) / (np.sqrt(self.var) + 1e-8)


class DQN(nn.Module):
    def __init__(self, input_dim, output_dim, hidden=128, use_layernorm=True):
        super().__init__()
        self.use_ln = use_layernorm

        self.feature = nn.Sequential(
            nn.Linear(input_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU()
        )

        if self.use_ln:
            # one layernorm for features
            self.ln = nn.LayerNorm(hidden)
        else:
            self.ln = None

        # dueling streams
        self.value_stream = nn.Sequential(
            nn.Linear(hidden, hidden//2),
            nn.ReLU(),
            nn.Linear(hidden//2, 1)
        )
        self.adv_stream = nn.Sequential(
            nn.Linear(hidden, hidden//2),
            nn.ReLU(),
            nn.Linear(hidden//2, output_dim)
        )

        # weight init
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.orthogonal_(m.weight, gain=np.sqrt(2))
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0.0)

    def forward(self, x):
        x = x.float()
        f = self.feature(x)
        if self.ln is not None:
            f = self.ln(f)
        value = self.value_stream(f)
        adv = self.adv_stream(f)
        # dueling combination
        q = value + (adv - adv.mean(dim=1, keepdim=True))
        return q

class Fighter:
    def __init__(self, player, x, y, flip, data, sprite_sheet, animation_steps, attack_sound, screen_width, role, training_phase, continue_from_episode = 0, mode="train"):
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
        self.mode = mode
        self.wins = 0
        self.smoothed_reward = 0.0

        # differentiate roles
        self.role = role
        self.training_phase = training_phase

        # DQN setup
        self.device            = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.state_dim         = 6
        self.action_space      = 5
        self.gamma             = 0.99
        self.batch_size        = 64
        self.lr                = 1e-4
        self.epsilon           = 1.0
        self.epsilon_min       = 0.1
        self.epsilon_decay     = 0.9995
        self.epsilon_linear_end = 0.05
        self.epsilon_anneal_episodes = 1500  # reach final eps by N episodes
        self.current_episode = 0
        self.memory            = deque(maxlen=100000)
        self.train_start       = 2000
        self.update_target_steps = 200
        self.step_count        = 0
        if self.mode=="play":
            self.epsilon = self.epsilon_linear_end

        # if self.training_phase==1 or self.role=="enemy":

        self.policy_net = DQN(self.state_dim, self.action_space).to(self.device)
        self.target_net = DQN(self.state_dim, self.action_space).to(self.device)
        
        npc_model_path = None
        npc_optim_path = None

        npc_model_path = None
        npc_optim_path = None

        if self.role == "enemy":
            npc_model_path = f"weights/player_2/phase_{self.training_phase-1}/model/_ep_{continue_from_episode}.pth"
            npc_optim_path = f"weights/player_2/phase_{self.training_phase-1}/optimizer/_ep_{continue_from_episode}.pth"
        else:
            npc_model_path = f"weights/player_1/phase_1/model/_ep_{continue_from_episode}.pth"
            npc_optim_path = f"weights/player_1/phase_1/optimizer/_ep_{continue_from_episode}.pth"

        print("path", npc_model_path, npc_optim_path)

        # always create optimizer first, then try to load states if available
        self.optimizer = optim.Adam(self.policy_net.parameters(),
                                     lr=self.lr,
                                     betas=(0.9, 0.999),
                                     eps=1e-08,
                                     weight_decay=0.0)

        if os.path.exists(npc_model_path):
            try:
                self.policy_net.load_state_dict(torch.load(npc_model_path, map_location=self.device))
                print("[INFO] Loaded NPC policy weights from Phase 1")
            except Exception as e:
                print(f"[WARN] Failed to load policy weights from {npc_model_path}: {e}")

            if os.path.exists(npc_optim_path):
                try:
                    self.optimizer.load_state_dict(torch.load(npc_optim_path, map_location=self.device))
                    print(f"[INFO] Loaded {self.role} optimizer state from Phase 1")
                except Exception as e:
                    print(f"[WARN] Failed to load optimizer state from {npc_optim_path}: {e}")
        else:
            print(f"[WARN] {self.role} policy checkpoint not found at {npc_model_path}. Starting from scratch.")

        
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()

        if (self.training_phase==2 or self.mode=="play") and self.role=="player":
            self.optimizer = None


    def anneal_epsilon(self):
        self.epsilon = max(
            self.epsilon_linear_end,
            1 - (self.current_episode / self.epsilon_anneal_episodes) * (1 - self.epsilon_linear_end)
        )


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
        # epsilon-greedy
        if random.random() < self.epsilon:
            return random.randrange(self.action_space)

        # normalize state to match training inputs
        state_arr = np.array(state, dtype=np.float32)
        if hasattr(self, "_rms"):
            try:
                state_arr = self._rms.normalize(state_arr)
            except Exception:
                pass

        state_v = torch.tensor(state_arr, dtype=torch.float32, device=self.device).unsqueeze(0)
        with torch.no_grad():
            qvals = self.policy_net(state_v)
            if torch.isnan(qvals).any():
                # fallback to random action on NaN
                return random.randrange(self.action_space)
        return int(qvals.argmax().cpu().numpy())


    def remember(self, s, a, r, s2, done):
        self.memory.append((s, a, r, s2, done))

    def optimize(self, use_double=True, tau=0.005, hard_update=False):
        """
        call every step or from training loop
        if hard_update True, target is copied every update_target_steps as before
        tau controls soft update when hard_update is False
        use_double toggles Double DQN target selection
        """
        if len(self.memory) < max(self.train_start, self.batch_size):
            if not hasattr(self, "_warned_replay_fill"):
                print(f"[INFO] {self.role} replay buffer warming up: {len(self.memory)}/{self.train_start}")
                self._warned_replay_fill = True
            return None, None

        batch = random.sample(self.memory, self.batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)

        states_arr      = np.array(states, dtype=np.float32)
        next_states_arr = np.array(next_states, dtype=np.float32)

        # update running stats and normalize
        if not hasattr(self, "_rms"):
            self._rms = RunningMeanStd(shape=states_arr.shape[1])
        self._rms.update(states_arr)
        states_arr = self._rms.normalize(states_arr)
        next_states_arr = self._rms.normalize(next_states_arr)

        states_v      = torch.from_numpy(states_arr).to(self.device)
        next_states_v = torch.from_numpy(next_states_arr).to(self.device)
        actions_v     = torch.tensor(actions, dtype=torch.long, device=self.device).unsqueeze(1)
        rewards_v     = torch.tensor(rewards, dtype=torch.float32, device=self.device).unsqueeze(1)
        dones_v       = torch.tensor(dones, dtype=torch.float32, device=self.device).unsqueeze(1)

        q_pred = self.policy_net(states_v).gather(1, actions_v)

        with torch.no_grad():
            if use_double:
                # Double DQN: select using policy_net, evaluate with target_net
                next_q_policy = self.policy_net(next_states_v)
                next_actions = next_q_policy.argmax(dim=1, keepdim=True)
                next_q_target = self.target_net(next_states_v).gather(1, next_actions)
            else:
                next_q_target = self.target_net(next_states_v).max(1)[0].unsqueeze(1)

            q_target = rewards_v + (1.0 - dones_v) * (self.gamma * next_q_target)

        loss_fn = nn.SmoothL1Loss()
        loss = loss_fn(q_pred, q_target)

        if self.optimizer is None:
            return None, None

        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.policy_net.parameters(), max_norm=10.0)
        self.optimizer.step()

        # soft update target
        if not hard_update:
            for tp, pp in zip(self.target_net.parameters(), self.policy_net.parameters()):
                tp.data.copy_(tp.data * (1.0 - tau) + pp.data * tau)

        if self.step_count % self.update_target_steps == 0:
            print(f"[INFO] Target network updated for {self.role} at step {self.step_count}")

        avg_q_pred = q_pred.mean().item()
        return float(loss.item()), avg_q_pred


    # def move(self, other, round_over, elapsed):
        if round_over:
            return
        reward = 0.0
        # auto face opponent
        self.flip = other.rect.x < self.rect.x

        state  = self.get_state(other)
        # action = self.select_action(state)
        done   = False
        prev_dx = abs(self.rect.x - other.rect.x)

        SPEED, GRAVITY = 10, 2
        dx, dy = 0, 0
        self.vel_y += GRAVITY
        dy       += self.vel_y

        # ---------------------------
        # HUMAN CONTROL SECTION
        # ---------------------------
        if self.mode == "play" and self.role == "player":
            keys = pygame.key.get_pressed()
            if keys[pygame.K_a]:
                dx = -SPEED
            elif keys[pygame.K_d]:
                dx = SPEED
            if keys[pygame.K_w] and not self.jump:
                self.vel_y = -30
                self.jump = True
            if keys[pygame.K_j] and self.attack_cooldown == 0:
                self.attacking = True
                # self.attack_sound.play()
                rect = (pygame.Rect(self.rect.right, self.rect.y, 3*self.rect.width, self.rect.height)
                        if not self.flip else
                        pygame.Rect(self.rect.x - 3*self.rect.width, self.rect.y, 3*self.rect.width, self.rect.height))
                if rect.colliderect(other.rect):
                    other.health -= 10/3
                self.attack_cooldown = 20
        # ---------------------------
        # AI CONTROL SECTION
        # ---------------------------
        else:
            action = self.select_action(state)
            if action == 0:
                dx = -SPEED
            elif action == 1:
                dx = SPEED
            elif action == 2 and not self.jump:
                self.vel_y = -30
                self.jump = True
            elif action in (3, 4) and self.attack_cooldown == 0:
                self.attacking = True
                rect = (pygame.Rect(self.rect.right, self.rect.y, 3*self.rect.width, self.rect.height)
                        if not self.flip else
                        pygame.Rect(self.rect.x - 3*self.rect.width, self.rect.y, 3*self.rect.width, self.rect.height))
                
                # other.health -= 10/3
                self.attack_cooldown = 20

                # 2. Attack result
                if rect.colliderect(other.rect):
                    other.health -= 10 / 3
                    if self.training_phase==1:
                        reward += 1.0
                    else:
                        reward += 0.3
                else:
                    if action in (3, 4):
                        if new_dx < 80:
                            reward += 0.1
                        else:  
                            reward -= 0.5  # stronger penalty for missing

            # -------------------------------------
            # REWARD CALCULATION (refactored)
            # -------------------------------------

            

            # 1. Distance shaping (encourage approaching, penalize retreat)
            new_dx = abs(self.rect.x - other.rect.x)
            if prev_dx is not None:
                move_toward = (prev_dx - new_dx) / 10.0  # scaled for stability
                reward += np.clip(move_toward, -0.5, 0.5)

            
            
            if not self.training_phase == 1:
                # 3. Balance reward (avoid one-sided fights)
                health_gap = abs(self.health - other.health) / 100.0
                balance_penalty = -min(1.0, (health_gap / 0.3) ** 2)
                reward += balance_penalty

                # 4. Duration shaping (penalize too-short or too-long fights)    
                target = 60.0
                sigma = 20.0
                reward += np.exp(-((elapsed - target) ** 2) / (2 * sigma ** 2)) - 0.5

                # 5. Terminal bonus
                if self.health <= 0 or other.health <= 0:
                    if 55 <= elapsed <= 65 and abs(self.health - other.health) < 25:
                        reward += 3.0
                    else:
                        reward -= 3.0

            # 6. Clamp and accumulate
            self.smoothed_reward = 0.9 * getattr(self, "smoothed_reward", 0) + 0.1 * reward
            reward = np.clip(self.smoothed_reward, -5.0, 5.0)
            self.episode_reward += reward

            # store debugging info (inspect these logs to tune coefficients)
            self.debug_last_reward = {
                "reward_total": float(reward),
                "distance": float(move_toward) if 'move_toward' in locals() else 0.0,
                "on_hit": float(1.0 if rect.colliderect(other.rect) else -0.5),
                "balance_penalty": float(balance_penalty) if 'balance_penalty' in locals() else 0.0,
            }


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
        if self.mode == "play" and self.role == "player":
            pass
        else:
            # self.update_action(action)
            self.remember(state, action, self.smoothed_reward, next_state, done)
        

        self.step_count += 1
        self.attack_cooldown = max(0, self.attack_cooldown - 1)


    def move(self, other, round_over, elapsed):
        if round_over:
            return

        # Auto face opponent
        self.flip = other.rect.x < self.rect.x

        # --- State setup ---
        state = self.get_state(other)
        done = False
        prev_dx = abs(self.rect.x - other.rect.x)

        # Physics constants
        SPEED, GRAVITY = 10, 2
        dx, dy = 0, 0
        self.vel_y += GRAVITY
        dy += self.vel_y

        # ======================
        #  HUMAN CONTROL SECTION
        # ======================
        if self.mode == "play" and self.role == "player":
            keys = pygame.key.get_pressed()
            if keys[pygame.K_a]:
                dx = -SPEED
            elif keys[pygame.K_d]:
                dx = SPEED

            if keys[pygame.K_w] and not self.jump:
                self.vel_y = -30
                self.jump = True

            if keys[pygame.K_j] and self.attack_cooldown == 0:
                self.attacking = True
                rect = (pygame.Rect(self.rect.right, self.rect.y, 3*self.rect.width, self.rect.height)
                        if not self.flip else
                        pygame.Rect(self.rect.x - 3*self.rect.width, self.rect.y, 3*self.rect.width, self.rect.height))
                if rect.colliderect(other.rect):
                    other.health -= 10 / 3
                self.attack_cooldown = 20

            # No learning during human control
            return

        # ==================
        #  AI CONTROL SECTION
        # ==================
        action = self.select_action(state)
        if action == 0:
            dx = -SPEED
        elif action == 1:
            dx = SPEED
        elif action == 2 and not self.jump:
            self.vel_y = -30
            self.jump = True
        elif action in (3, 4) and self.attack_cooldown == 0:
            self.attacking = True
            rect = (pygame.Rect(self.rect.right, self.rect.y, 3*self.rect.width, self.rect.height)
                    if not self.flip else
                    pygame.Rect(self.rect.x - 3*self.rect.width, self.rect.y, 3*self.rect.width, self.rect.height))
            self.attack_cooldown = 20

            # Apply hit
            if rect.colliderect(other.rect):
                other.health -= 10 / 3

        # Prevent overlap
        new_x = self.rect.x + dx
        temp = self.rect.copy()
        temp.x = new_x
        if temp.colliderect(other.rect):
            dx = 0

        # Apply position
        self.rect.x = max(0, min(self.rect.x + dx, self.screen_width - self.rect.width))
        self.rect.y = max(0, min(self.rect.y + dy, (self.screen_width / 2) - self.rect.height))
        if self.rect.y >= (self.screen_width / 2) - self.rect.height:
            self.jump = False

        if self.health <= 0:
            done = True
            self.alive = False

        # Next state
        new_dx = abs(self.rect.x - other.rect.x)
        next_state = self.get_state(other)

        # ==========================
        #  REWARD CALCULATION PHASE
        # ==========================
        if self.training_phase == 1:
            reward, debug_info = self.compute_reward_phase1(other, prev_dx, new_dx, elapsed, action)
        else:
            reward, debug_info = self.compute_reward_phase2(other, prev_dx, new_dx, elapsed, action)

        # Apply smoothing & clipping
        self.smoothed_reward = 0.9 * getattr(self, "smoothed_reward", 0) + 0.1 * reward
        reward = np.clip(self.smoothed_reward, -5.0, 5.0)
        self.episode_reward += reward

        # Memory update
        self.remember(state, action, reward, next_state, done)

        # Logging for tuning
        self.debug_last_reward = debug_info

        # Step bookkeeping
        self.step_count += 1
        self.attack_cooldown = max(0, self.attack_cooldown - 1)

    def compute_reward_phase1(self, other, prev_dx, new_dx, elapsed, action):
        """
        Simple reward shaping for Phase 1:
        - Encourages moving toward the opponent
        - Rewards successful hits
        - Penalizes whiffed attacks or running away
        - Adds small survival/time stability term
        """

        reward = 0.0
        debug = {}

        # 1. Distance shaping — get closer = good, retreat = bad
        if prev_dx is not None:
            move_toward = (prev_dx - new_dx) / 10.0
            reward += np.clip(move_toward, -0.3, 0.3)
            debug["move_toward"] = float(np.clip(move_toward, -0.3, 0.3))

        # 2. Attack reward — hit = +1, miss = -0.5
        if action in (3, 4) and self.attack_cooldown == 19:  # just attacked
            rect = (pygame.Rect(self.rect.right, self.rect.y, 3*self.rect.width, self.rect.height)
                    if not self.flip else
                    pygame.Rect(self.rect.x - 3*self.rect.width, self.rect.y, 3*self.rect.width, self.rect.height))
            if rect.colliderect(other.rect):
                reward += 1.0
                debug["on_hit"] = 1.0
            else:
                reward -= 0.5
                debug["on_hit"] = -0.5

        # 3. Time survival shaping — small reward for staying alive
        reward += 0.01
        debug["survival"] = 0.01

        # 4. Terminal bonus — win/loss outcome
        if self.health <= 0 or other.health <= 0:
            if self.health > other.health:
                reward += 2.0
                debug["terminal"] = 2.0
            else:
                reward -= 2.0
                debug["terminal"] = -2.0

        # Clip and return
        reward = np.clip(reward, -5.0, 5.0)
        debug["reward_total"] = float(reward)

        return reward, debug

    
    def compute_reward_phase2(self, other, prev_dx, new_dx, elapsed, action):
        """
        Adaptive/human-like reward for Phase 2:
        - Still values competence (move, hit)
        - Adds balance & pacing: discourages one-sided wins or ultra-short fights
        - Encourages 'human' behaviours like spacing, counter-timing, mercy when opponent weak
        """

        reward = 0.0
        debug = {}

        # 1. Distance shaping – prefer keeping tactical spacing, not constant rush
        optimal_min, optimal_max = 80, 250
        if new_dx < optimal_min:
            dist_reward = -((optimal_min - new_dx) / optimal_min) * 0.3  # too close
        elif new_dx > optimal_max:
            dist_reward = -((new_dx - optimal_max) / optimal_max) * 0.3  # too far
        else:
            dist_reward = 0.2  # good range
        reward += dist_reward
        debug["distance"] = float(dist_reward)

        # 2. Movement direction – backing off after opponent attack = okay
        if prev_dx is not None:
            move_toward = (prev_dx - new_dx) / 10.0
            reward += np.clip(move_toward, -0.2, 0.2)
            debug["move_toward"] = float(np.clip(move_toward, -0.2, 0.2))

        # 3. Attack behaviour – success, restraint, or tactical retreat
        if action in (3, 4) and self.attack_cooldown == 19:
            rect = (pygame.Rect(self.rect.right, self.rect.y, 3*self.rect.width, self.rect.height)
                    if not self.flip else
                    pygame.Rect(self.rect.x - 3*self.rect.width, self.rect.y, 3*self.rect.width, self.rect.height))
            if rect.colliderect(other.rect):
                reward += 0.5
                debug["on_hit"] = 0.5
            else:
                reward -= 0.3
                debug["on_hit"] = -0.3

        # 4. Balance shaping – keep fights close and fair
        health_gap = abs(self.health - other.health) / 100.0
        balance_penalty = -min(1.0, (health_gap / 0.3) ** 2)
        reward += balance_penalty
        debug["balance_penalty"] = float(balance_penalty)

        # 5. Mercy / taunt behaviour – small reward if leading but holding back
        # if self.health > other.health and new_dx > optimal_min and self.attack_cooldown > 0:
        if self.health > other.health and balance_penalty < -0.2:
            reward += 0.1  # light reward for not pressing advantage too hard
            debug["mercy"] = 0.1

        # 6. Duration shaping – prefer medium-length fights (not too short/long)
        target, sigma = 60.0, 20.0
        duration_bonus = np.exp(-((elapsed - target) ** 2) / (2 * sigma ** 2)) - 0.5
        reward += 1.5 * duration_bonus
        debug["duration"] = float(1.5 * duration_bonus)

        # 7. Terminal conditions
        if self.health <= 0 or other.health <= 0:
            if 55 <= elapsed <= 65 and abs(self.health - other.health) < 25:
                terminal = 1.5
            else:
                terminal = -1.5
            reward += terminal
            debug["terminal"] = float(terminal)

        # Final smoothing & clipping
        # reward = np.clip(reward, -5.0, 5.0)
        debug["reward_total"] = float(reward)

        return reward, debug


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
        self.debug_last_reward = None
