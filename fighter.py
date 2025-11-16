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
        self.jump_cooldown   = 0

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
        if self.training_phase ==1:
            self.epsilon_anneal_episodes = 800  # reach final eps by N episodes
        else:
            self.epsilon_anneal_episodes = 1500
        self.current_episode = 0
        self.memory            = deque(maxlen=100000)
        self.train_start       = 2000
        self.update_target_steps = 200
        self.step_count        = 0
        self.action_timer = 0
        if self.mode=="play":
            self.epsilon = 0

        # if self.training_phase==1 or self.role=="enemy":

        self.policy_net = DQN(self.state_dim, self.action_space).to(self.device)
        self.target_net = DQN(self.state_dim, self.action_space).to(self.device)
        
        npc_model_path = None
        npc_optim_path = None

        npc_model_path = None
        npc_optim_path = None

        if self.role == "enemy":
            npc_model_path = f"weights/player_2/phase_{self.training_phase}/model/_ep_{continue_from_episode}.pth"
            npc_optim_path = f"weights/player_2/phase_{self.training_phase}/optimizer/_ep_{continue_from_episode}.pth"
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
        # self.epsilon = max(
        #     self.epsilon_linear_end,
        #     1 - (self.current_episode / self.epsilon_anneal_episodes) * (1 - self.epsilon_linear_end)
        # )

        # two-stage linear schedule: 1.0 -> 0.1 over first 40% episodes,
        # then 0.1 -> final_eps over next 60%
        final_eps = 0.02   # or 0.05 if you prefer slightly higher exploration
        halfpoint = int(0.4 * max(1, self.epsilon_anneal_episodes))
        if self.current_episode >= self.epsilon_anneal_episodes:
            self.epsilon = final_eps
        else:
            if self.current_episode <= halfpoint:
                # stage 1: 1.0 -> 0.1
                t = self.current_episode / max(1, halfpoint)
                self.epsilon = 1.0 + t * (0.1 - 1.0)
            else:
                # stage 2: 0.1 -> final_eps
                t = (self.current_episode - halfpoint) / max(1, self.epsilon_anneal_episodes - halfpoint)
                self.epsilon = 0.1 + t * (final_eps - 0.1)



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
        # if round_over:
        #     return
        # reward = 0.0
        # # auto face opponent
        # self.flip = other.rect.x < self.rect.x

        # state  = self.get_state(other)
        # # action = self.select_action(state)
        # done   = False
        # prev_dx = abs(self.rect.x - other.rect.x)

        # SPEED, GRAVITY = 10, 2
        # dx, dy = 0, 0
        # self.vel_y += GRAVITY
        # dy       += self.vel_y

        # # ---------------------------
        # # HUMAN CONTROL SECTION
        # # ---------------------------
        # if self.mode == "play" and self.role == "player":
        #     keys = pygame.key.get_pressed()
        #     if keys[pygame.K_a]:
        #         dx = -SPEED
        #     elif keys[pygame.K_d]:
        #         dx = SPEED
        #     if keys[pygame.K_w] and not self.jump:
        #         self.vel_y = -30
        #         self.jump = True
        #     if keys[pygame.K_j] and self.attack_cooldown == 0:
        #         self.attacking = True
        #         # self.attack_sound.play()
        #         rect = (pygame.Rect(self.rect.right, self.rect.y, 3*self.rect.width, self.rect.height)
        #                 if not self.flip else
        #                 pygame.Rect(self.rect.x - 3*self.rect.width, self.rect.y, 3*self.rect.width, self.rect.height))
        #         if rect.colliderect(other.rect):
        #             other.health -= 10/3
        #         self.attack_cooldown = 20
        # # ---------------------------
        # # AI CONTROL SECTION
        # # ---------------------------
        # else:
        #     action = self.select_action(state)
        #     if action == 0:
        #         dx = -SPEED
        #     elif action == 1:
        #         dx = SPEED
        #     elif action == 2 and not self.jump:
        #         self.vel_y = -30
        #         self.jump = True
        #     elif action in (3, 4) and self.attack_cooldown == 0:
        #         self.attacking = True
        #         rect = (pygame.Rect(self.rect.right, self.rect.y, 3*self.rect.width, self.rect.height)
        #                 if not self.flip else
        #                 pygame.Rect(self.rect.x - 3*self.rect.width, self.rect.y, 3*self.rect.width, self.rect.height))
                
        #         # other.health -= 10/3
        #         self.attack_cooldown = 20

        #         # 2. Attack result
        #         if rect.colliderect(other.rect):
        #             other.health -= 10 / 3
        #             if self.training_phase==1:
        #                 reward += 1.0
        #             else:
        #                 reward += 0.3
        #         else:
        #             if action in (3, 4):
        #                 if new_dx < 80:
        #                     reward += 0.1
        #                 else:  
        #                     reward -= 0.5  # stronger penalty for missing

        #     # -------------------------------------
        #     # REWARD CALCULATION (refactored)
        #     # -------------------------------------

            

        #     # 1. Distance shaping (encourage approaching, penalize retreat)
        #     new_dx = abs(self.rect.x - other.rect.x)
        #     if prev_dx is not None:
        #         move_toward = (prev_dx - new_dx) / 10.0  # scaled for stability
        #         reward += np.clip(move_toward, -0.5, 0.5)

            
            
        #     if not self.training_phase == 1:
        #         # 3. Balance reward (avoid one-sided fights)
        #         health_gap = abs(self.health - other.health) / 100.0
        #         balance_penalty = -min(1.0, (health_gap / 0.3) ** 2)
        #         reward += balance_penalty

        #         # 4. Duration shaping (penalize too-short or too-long fights)    
        #         target = 60.0
        #         sigma = 20.0
        #         reward += np.exp(-((elapsed - target) ** 2) / (2 * sigma ** 2)) - 0.5

        #         # 5. Terminal bonus
        #         if self.health <= 0 or other.health <= 0:
        #             if 55 <= elapsed <= 65 and abs(self.health - other.health) < 25:
        #                 reward += 3.0
        #             else:
        #                 reward -= 3.0

        #     # 6. Clamp and accumulate
        #     self.smoothed_reward = 0.9 * getattr(self, "smoothed_reward", 0) + 0.1 * reward
        #     reward = np.clip(self.smoothed_reward, -5.0, 5.0)
        #     self.episode_reward += reward

        #     # store debugging info (inspect these logs to tune coefficients)
        #     self.debug_last_reward = {
        #         "reward_total": float(reward),
        #         "distance": float(move_toward) if 'move_toward' in locals() else 0.0,
        #         "on_hit": float(1.0 if rect.colliderect(other.rect) else -0.5),
        #         "balance_penalty": float(balance_penalty) if 'balance_penalty' in locals() else 0.0,
        #     }


        # # block overlap
        # new_x = self.rect.x + dx
        # temp = self.rect.copy()
        # temp.x = new_x
        # if temp.colliderect(other.rect): dx = 0

        # self.rect.x = max(0, min(self.rect.x + dx, self.screen_width - self.rect.width))
        # self.rect.y = max(0, min(self.rect.y + dy, (self.screen_width/2) - self.rect.height))
        # if self.rect.y >= (self.screen_width/2) - self.rect.height:
        #     self.jump = False

        # if self.health <= 0:
        #     done = True
        #     self.alive = False

        # next_state = self.get_state(other)
        # if self.mode == "play" and self.role == "player":
        #     pass
        # else:
        #     # self.update_action(action)
        #     self.remember(state, action, self.smoothed_reward, next_state, done)
        

        # self.step_count += 1
        # self.attack_cooldown = max(0, self.attack_cooldown - 1)

    def get_action(self, other):
        """
        Scripted enemy policy for demo:
        - persistent movements (hold for several frames)
        - aggression scales with opponent health (more aggressive when opponent has lots of health)
        - retreats/mercy when opponent is low and self is high
        - jitter, feints, occasional jumps
        - returns integer action (0:left, 1:right, 2:jump, 3:attack_primary, 4:attack_alt)
        """

        # sanity init of bookkeeping (if constructor wasn't updated)
        if not hasattr(self, "script_action_hold"):
            self.script_action_hold = None
            self.script_hold_timer = 0
            self.action_timer = 0
            self.last_attack_frame = -9999
            self.feint_pending = False

        self.action_timer += 1
        # decrease any hold timers
        if self.script_hold_timer > 0:
            self.script_hold_timer -= 1
            # still holding an action: return it
            if self.script_hold_timer > 0:
                return self.script_action_hold
            # if timer just hit 0 we will fall through and pick a new action

        # short names
        my_hp = float(self.health)
        their_hp = float(other.health)
        abs_dist = abs(self.rect.x - other.rect.x)        # pixel distance
        sign = 1 if (other.rect.x > self.rect.x) else -1  # direction to move to approach (other is to right => sign=1)

        # behaviour knobs
        desired_range = 150           # ideal spacing in pixels (we'll try to maintain)
        close_range   = 90            # "in attack range" threshold
        far_range     = 300

        # Aggression score:
        # - increases when opponent has high health
        # - reduced when self is very healthy (we may be conservative/taunt)
        # - reduced further when opponent is low and we are healthy (mercy)
        opp_norm = np.clip(their_hp / 100.0, 0.0, 1.0)
        self_norm = np.clip(my_hp / 100.0, 0.0, 1.0)

        # base aggression (0..1)
        agg = 0.3 + 0.7 * opp_norm          # baseline increases strongly with opponent health
        # reduce aggression if we have a big lead
        health_gap = (self_norm - opp_norm)    # positive means we are healthier
        if health_gap > 0.25:
            agg -= health_gap * 0.8           # back off when we have a clear advantage
        agg = float(np.clip(agg, 0.05, 0.95))
        if health_gap < -0.25:
            agg -= health_gap * 0.8           # get aggressive if about to lose
        agg = float(np.clip(agg, 0.05, 0.95))

        # If opponent is very low and we are much stronger -> prefer retreat / taunt
        if their_hp < 25 and my_hp > 55 and abs_dist < desired_range * 1.6:
            # retreat quickly and do taunts/feints
            # move away (opponent is on our right => move right to retreat)
            retreat_dir = 0 if (self.rect.x > other.rect.x) else 1
            # hold retreat for a short burst
            self.script_action_hold = retreat_dir
            self.script_hold_timer = random.randint(8, 18)
            # occasional taunt: sometimes trigger a jump or small forward step instead of pure retreat
            if random.random() < 0.25:
                # 2 = jump (small taunt)
                self.feint_pending = True
            return self.script_action_hold

        # If we are much weaker -> become desperate & aggressive
        if my_hp < 30 and their_hp > 50:
            # DESPERATION MODE: rush in aggressively
            # Move toward the player for several frames
            direction = 1 if (self.rect.x < other.rect.x) else 0
            self.script_action_hold = direction
            self.script_hold_timer = random.randint(6, 12)

            # 30–40% chance to perform an immediate desperate attack
            if abs_dist < desired_range * 1.2 and random.random() < 0.4:
                self.script_action_hold = 3  # attack
                self.script_hold_timer = random.randint(4, 8)
                return 3

            # Occasional erratic jumps (feels panic/desperation)
            if random.random() < 0.1:
                self.script_action_hold = 2  # jump
                self.script_hold_timer = random.randint(4, 8)
                return 2

        # Normal behaviour: maintain spacing, close to attack, then retreat back to desired range
        # If we are too far -> approach
        if abs_dist > desired_range + 30:
            # approach (move toward opponent)
            approach_dir = 1 if (other.rect.x > self.rect.x) else 0
            self.script_action_hold = approach_dir
            self.script_hold_timer = random.randint(6, 14)   # hold movement for a handful of frames (makes motion smooth)
            # small chance of a feint while approaching
            if random.random() < 0.08:
                # do a small jump as a feint
                self.feint_pending = True
            return self.script_action_hold

        # If in comfortable range, consider attacking with probability proportional to aggression
        if abs_dist <= desired_range + 10:
            # base attack probability scaled by aggression and closeness
            dist_factor = np.clip(1.0 - (abs_dist / (desired_range + 10.0)), 0.0, 1.0)  # 1 when very close
            attack_prob = 0.08 + 0.45 * agg * dist_factor   # tuned probabilities
            # don't spam attacks: require a minimum gap between attack attempts
            min_frames_between_attacks = 18
            can_attack = (self.action_timer - self.last_attack_frame) > min_frames_between_attacks

            if can_attack and random.random() < attack_prob:
                # choose primary vs alternate attack (small randomness)
                attack_action = 3 if random.random() < 0.8 else 4
                self.script_action_hold = attack_action
                # attack animation / execution should be held for a bit so it's visible
                self.script_hold_timer = random.randint(10, 18)
                self.last_attack_frame = self.action_timer
                # small chance to feint (attack animation but intentionally short/higher miss chance simulated by alternating with jump)
                if random.random() < 0.12:
                    self.feint_pending = True
                return attack_action

            # otherwise we prefer to "dance" at spacing: small jitter moves or idle (simulate taunting)
            if random.random() < 0.18:
                # small random step left/right for a short time (jitter)
                jitter_dir = 1 if random.random() < 0.5 else 0
                self.script_action_hold = jitter_dir
                self.script_hold_timer = random.randint(3, 8)
                return self.script_action_hold

            if random.random() < 0.07:
                # a little show-off: quick jump / feint
                self.script_action_hold = 2
                self.script_hold_timer = random.randint(4, 8)
                return 2

            # taunt: do nothing meaningful, but since there is no explicit idle action, we jitter tiny steps occasionally
            if random.random() < 0.25:
                # small step away to create spacing and 'taunt'
                away_dir = 0 if (self.rect.x > other.rect.x) else 1
                self.script_action_hold = away_dir
                self.script_hold_timer = random.randint(4, 10)
                return away_dir

            # fallback: small time where we choose to not change position (simulate "no-op")
            # there's no explicit no-op action, so return a tiny jitter (one frame) or keep previous action
            if self.script_action_hold is not None and self.script_hold_timer <= 0:
                # hold nothing — return a tiny random movement single-frame
                if random.random() < 0.5:
                    return 4  # a quick alternative attack (serves as a visual micro-action)
                return 1 if random.random() < 0.5 else 0

        # Catch-all: small random behavior to keep it lively.
        if random.random() < 0.08:
            # jump occasionally
            self.script_action_hold = 2
            self.script_hold_timer = random.randint(3, 7)
            return 2
        if random.random() < 0.12:
            # spontaneous small step
            self.script_action_hold = 1 if random.random() < 0.5 else 0
            self.script_hold_timer = random.randint(4, 10)
            return self.script_action_hold

        # final fallback: idle-ish micro-move
        return 4  # a benign quick-action (treated as alternate attack in engine but visually OK for demo)


    def get_new_script_action(self, other):
        """
        Aggressive, consistent fighter:
        - always tries to hit when in attack range
        - approaches quickly
        - fewer jumps and feints
        - mercy condition respected
        """

        # Init missing fields
        if not hasattr(self, "script_action_hold"):
            self.script_action_hold = None
            self.script_hold_timer = 0
            self.action_timer = 0
            self.last_attack_frame = -9999
            self.feint_pending = False

        self.action_timer += 1

        # Hold logic
        if self.script_hold_timer > 0:
            self.script_hold_timer -= 1
            if self.script_hold_timer > 0:
                return self.script_action_hold

        # Shortcuts
        my_hp = float(self.health)
        their_hp = float(other.health)
        abs_dist = abs(self.rect.x - other.rect.x)

        # Direction: move toward opponent (1=right, 0=left)
        towards = 1 if (other.rect.x > self.rect.x) else 0
        away    = 0 if (other.rect.x > self.rect.x) else 1

        # Tuned ranges
        desired_range = 150
        close_range   = 90

        # Aggression based heavily on opponent health
        opp_norm = np.clip(their_hp / 100.0, 0.0, 1.0)
        agg = 0.55 + 0.45 * opp_norm  # always high
        agg = float(np.clip(agg, 0.65, 0.98))

        # -------------------------------------------------------------------
        #  MERCY CONDITION (no attacking)
        # -------------------------------------------------------------------
        if their_hp < 25 and my_hp > 55 and abs_dist < desired_range * 1.5:
            # retreat smoothly
            self.script_action_hold = away
            self.script_hold_timer = random.randint(10, 18)
            return away

        # -------------------------------------------------------------------
        #  DESPERATION MODE (even more aggressive)
        # -------------------------------------------------------------------
        if my_hp < 30 and their_hp > 50:
            # rush
            self.script_action_hold = towards
            self.script_hold_timer = random.randint(6, 12)

            # if in attack range, attack immediately
            if abs_dist <= desired_range:
                self.last_attack_frame = self.action_timer
                return 3

            return towards

        # -------------------------------------------------------------------
        #  AGGRESSIVE SPACING LOGIC
        # -------------------------------------------------------------------

        # Too far: sprint in
        if abs_dist > desired_range:
            self.script_action_hold = towards
            self.script_hold_timer = random.randint(8, 14)
            return towards

        # Slightly out of ideal close range: close to pressure
        if abs_dist > close_range:
            self.script_action_hold = towards
            self.script_hold_timer = random.randint(6, 10)
            return towards

        # -------------------------------------------------------------------
        #  IN RANGE → ALWAYS TRY TO HIT
        # -------------------------------------------------------------------

        min_frames_between_attacks = 16
        can_attack = (self.action_timer - self.last_attack_frame) > min_frames_between_attacks

        if can_attack:
            # attack aggressively
            attack_action = 3 if random.random() < 0.85 else 4
            self.script_action_hold = attack_action
            self.script_hold_timer = random.randint(12, 16)
            self.last_attack_frame = self.action_timer
            return attack_action

        # Otherwise maintain close pressure
        # gentle micro steps to keep in range
        if abs_dist < close_range - 20:
            # back up slightly to avoid hugging too close
            self.script_action_hold = away
            self.script_hold_timer = random.randint(4, 8)
            return away

        # Otherwise hold ground
        return towards



    def get_dumb_action(self, other):
        """
        Worse/jittery version of scripted enemy:
        - more hesitation
        - shorter decision holds
        - random stutters and feints
        - slight irrational decisions
        """

        # sanity init
        if not hasattr(self, "script_action_hold"):
            self.script_action_hold = None
            self.script_hold_timer = 0
            self.action_timer = 0
            self.last_attack_frame = -9999
            self.feint_pending = False

        self.action_timer += 1

        # --- More chance to break out early from a hold (simulate sloppiness)
        if self.script_hold_timer > 0:
            self.script_hold_timer -= 1

            # enemy may prematurely cancel its plan
            if random.random() < 0.10:
                self.script_hold_timer = 0  

            if self.script_hold_timer > 0:
                return self.script_action_hold

        # short names
        my_hp = float(self.health)
        their_hp = float(other.health)
        abs_dist = abs(self.rect.x - other.rect.x)
        sign = 1 if (other.rect.x > self.rect.x) else -1

        desired_range = 150
        close_range = 90
        far_range = 300

        opp_norm = np.clip(their_hp / 100.0, 0.0, 1.0)
        self_norm = np.clip(my_hp / 100.0, 0.0, 1.0)

        agg = 0.3 + 0.7 * opp_norm
        health_gap = (self_norm - opp_norm)
        if health_gap > 0.25:
            agg -= health_gap * 0.8
        agg = float(np.clip(agg, 0.05, 0.95))
        if health_gap < -0.25:
            agg -= health_gap * 0.8
        agg = float(np.clip(agg, 0.05, 0.95))

        # --- MORE RANDOM HESITATION applied globally ---
        if random.random() < 0.05:
            # 1–2 frame stutter
            return 1 if random.random() < 0.5 else 0

        if random.random() < 0.02:
            # pointless jump
            return 2

        # --- MERCY / TAUNT (same logic, but messier) ---
        if their_hp < 25 and my_hp > 55 and abs_dist < desired_range * 1.6:
            retreat_dir = 0 if (self.rect.x > other.rect.x) else 1
            self.script_action_hold = retreat_dir
            
            # shorter hold → jittery
            self.script_hold_timer = random.randint(4, 12)

            # extra taunt randomness
            if random.random() < 0.35:
                self.feint_pending = True
            if random.random() < 0.2:
                return 2  
            return retreat_dir

        # --- DESPERATION (but more sloppy) ---
        if my_hp < 30 and their_hp > 50:
            direction = 1 if (self.rect.x < other.rect.x) else 0
            self.script_action_hold = direction

            # shorter holds ⇒ panic jitter
            self.script_hold_timer = random.randint(4, 8)

            # more reckless jumps
            if random.random() < 0.18:
                return 2  

            # attack, but dumber
            if abs_dist < desired_range * 1.2 and random.random() < 0.35:
                self.script_action_hold = 3
                self.script_hold_timer = random.randint(6, 10)
                self.last_attack_frame = self.action_timer
                return 3

        # --- TOO FAR → approach, but more indecisive ---
        if abs_dist > desired_range + 30:
            approach_dir = 1 if (other.rect.x > self.rect.x) else 0

            # hesitation flip
            if random.random() < 0.12:
                approach_dir = 1 - approach_dir

            self.script_action_hold = approach_dir
            self.script_hold_timer = random.randint(4, 10)

            if random.random() < 0.12:
                self.feint_pending = True
            if random.random() < 0.05:
                return 2
            return approach_dir

        # --- In combat range: attack or jitter around ---
        if abs_dist <= desired_range + 10:
            dist_factor = np.clip(1.0 - (abs_dist / (desired_range + 10)), 0.0, 1.0)
            attack_prob = (0.06 + 0.35 * agg * dist_factor)  # slightly lower => worse

            can_attack = (self.action_timer - self.last_attack_frame) > 20

            # BAD DECISION: sometimes choose wrong direction instead of attack
            if random.random() < 0.06:
                wrong = 1 if random.random() < 0.5 else 0
                return wrong

            # Attack
            if can_attack and random.random() < attack_prob:
                attack_action = 3 if random.random() < 0.75 else 4  

                # sometimes fail and do a pointless stance
                if random.random() < 0.15:
                    return 2  

                self.script_action_hold = attack_action
                self.script_hold_timer = random.randint(8, 14)
                self.last_attack_frame = self.action_timer

                if random.random() < 0.15:
                    self.feint_pending = True
                return attack_action

            # jitter around the range
            if random.random() < 0.25:
                jitter_dir = 1 if random.random() < 0.5 else 0
                self.script_action_hold = jitter_dir
                self.script_hold_timer = random.randint(2, 6)
                return jitter_dir

            if random.random() < 0.10:
                return 2

            if random.random() < 0.20:
                away_dir = 0 if (self.rect.x > other.rect.x) else 1
                return away_dir

            # no-op micro jitter
            return 1 if random.random() < 0.5 else 0

        # --- Catch-all random junk (worse) ---
        if random.random() < 0.12:
            return 2
        if random.random() < 0.15:
            self.script_action_hold = 1 if random.random() < 0.5 else 0
            self.script_hold_timer = random.randint(2, 6)
            return self.script_action_hold

        return 1 if random.random() < 0.5 else 0




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

        else:
            # ==================
            #  AI CONTROL SECTION
            # ==================
            if self.role == "scripted_smart":
                action = self.get_action(other)
            elif self.role == "scripted_dumb":
                action = self.get_dumb_action(other)
            elif self.role == "new_script":
                action = self.get_new_script_action(other)
            else:
                action = self.select_action(state)
            if action == 0:
                dx = -SPEED
            elif action == 1:
                dx = SPEED
            elif action == 2 and not self.jump and self.jump_cooldown == 0:
                self.vel_y = -30
                self.jump = True
                self.jump_cooldown = 25
            elif action == 3 and self.attack_cooldown == 0:
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

        if self.mode == "train" and self.role == "enemy":


            # ==========================
            #  REWARD CALCULATION PHASE
            # ==========================
            if self.training_phase == 1:
                reward, debug_info = self.compute_reward_phase1(other, prev_dx, new_dx, elapsed, action)
            elif self.training_phase == 2:
                reward, debug_info = self.compute_reward_phase2(other, prev_dx, new_dx, elapsed, action)
            else:
                reward, debug_info = self.compute_reward_phase3(other, prev_dx, new_dx, elapsed, action)

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
        self.jump_cooldown = max(0, self.jump_cooldown - 1)

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

    
    def compute_reward_phase3(self, other, prev_dx, new_dx, elapsed, action):
        """
        Phase 3 (rebalanced):
        - Spacing reward centered at 0 (Gaussian)
        - Small per-frame survival bonus
        - Bigger hit reward (sparse)
        - softer miss / jump penalties
        - motion shaping kept small
        """
        reward = 0.0
        debug = {}

        # 1) Gaussian spacing (centered around 0)
        optimal_dist = 150.0
        sigma = 60.0
        spacing = np.exp(-((new_dx - optimal_dist) ** 2) / (2 * sigma ** 2))
        # normalize to roughly [0,1] then remap to [-0.15, +0.25] (slightly positive bias)
        spacing_reward = spacing * 0.4 - 0.08
        reward += spacing_reward
        debug["spacing"] = float(spacing_reward)

        # 2) tiny survival bonus so episodes don't accumulate strong negative drift
        survival_bonus = 0.01
        reward += survival_bonus
        debug["survival"] = float(survival_bonus)

        # 3) movement direction shaping (small)
        if prev_dx is not None:
            delta = (prev_dx - new_dx) / 20.0
            delta = float(np.clip(delta, -0.12, 0.12))
            reward += delta
            debug["movement"] = delta

        # 4) attack reward on the action frame (sparse)
        # we check attack_cooldown trigger like before (assumes cooldown set to 19 when attack starts)
        if action in (3, 4) and self.attack_cooldown == 19:
            rect = (pygame.Rect(self.rect.right, self.rect.y, 3*self.rect.width, self.rect.height)
                    if not self.flip else
                    pygame.Rect(self.rect.x - 3*self.rect.width, self.rect.y, 3*self.rect.width, self.rect.height))
            if rect.colliderect(other.rect):
                hit_reward = 1.0   # stronger, less frequent positive
                reward += hit_reward
                debug["hit"] = hit_reward
            else:
                miss_penalty = -0.15
                reward += miss_penalty
                debug["hit"] = miss_penalty

        # 5) discourage needless jumps (smaller)
        if action == 2:
            reward -= 0.1
            debug["jump_penalty"] = -0.1

        # Clip and return debug
        reward = np.clip(reward, -3.0, 3.0)
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
