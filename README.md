# RL Fighter Technical Report - Content Mapping

## 1. Abstract (4–5 sentences)

* Purpose: Training an RL agent for adaptive NPC behavior in a 2D/3D fighting game.
* Method: Deep Q-Network (DQN) with epsilon-greedy policy, replay buffer, multi-phase training (Phase 1 → Phase 2 → Phase 3).
* Key findings: NPC learns to balance winning with human-like behavior; training stabilizes by ~2000 episodes.
* Evidence: Reward curves, win/loss statistics.

## 2. Background / Introduction (1–2 paragraphs)

* RL in games: trial-and-error learning, reward-based policy improvement.
* NPC adaptation: importance for realistic gameplay.
* References: BoxStacker for RL in virtual environments, general DQN literature.
* Optional: Brief mention of challenges (sparse rewards, balancing realism vs performance).

## 3. Methodology (~1.2–1.5 pages)

### A. Environment Setup

* Game environment description (2D/3D arena, player sprites).
* Phases:

  * Phase 1: baseline training against default opponent.
  * Phase 2: adaptive reward shaping for human-like behavior.
  * Phase 3: (if any) further strategy refinement.

### B. State Space (Table 1)

| Variable        | Description                                      |
| --------------- | ------------------------------------------------ |
| Player position | x, y coordinates of player                       |
| Velocity        | Speed in x/y directions                          |
| Health          | Current health/life points                       |
| Distance        | Distance to opponent                             |
| Cooldowns       | Time left for special moves                      |
| Others          | Optional: status effects, environment boundaries |

### C. Action Space (Table 2)

| Action          | Description                   |
| --------------- | ----------------------------- |
| Move left/right | Discrete horizontal movement  |
| Jump / Crouch   | Vertical movement             |
| Attack          | Basic attack                  |
| Block           | Defensive move                |
| Special move    | Any combo or signature attack |

### D. Reward Design

* Phase 2 reward function (`compute_reward_phase2`):

  * Positive reward for hitting opponent.
  * Negative reward for taking damage.
  * Bonus for winning a round.
  * Penalty for unrealistic/repetitive moves.
* Optionally include pseudocode or simplified formula as a figure.

### E. Algorithm

* DQN: Q-network, epsilon-greedy policy, replay buffer, learning rate.
* Mention phase-specific tweaks (epsilon decay, reward scaling).

### Figures/Tables:

* Figure 1: RL Agent–Environment Loop (state → action → reward → next state)
* Figure 2: Training workflow diagram (Phase 1 → Phase 2 → Phase 3)

## 4. Results (~0.5 page)

* Win/loss statistics: table or summary (Phase 1 vs Phase 2).
* Reward trends: plot of average reward vs episodes.
* Behavior adaptation: short description (e.g., NPC learned to block more frequently against strong attacks).
* Optional: screenshot of environment showing NPC action.

### Figures/Tables:

* Figure 3: Reward vs episodes curve
* Optional: screenshot of NPC in action

## 5. Discussion (~0.25 page)

* Strengths: adaptive NPC, learning stabilizes by ~2000 episodes, reward shaping effective.
* Challenges: sparse rewards, balancing optimal vs human-like behavior, training time.

## 6. Limitations & Future Work (~0.25 page)

* Limitations: simplified game environment, limited state/action space, potential overfitting.
* Future work:

  * Multi-agent training for more complex scenarios
  * Introduce continuous action space
  * Curiosity-based or intrinsic rewards
  * Expand environment complexity (more moves, obstacles, or physics)

## 7. References

- Baker, B., Akkaya, I., Zaremba, W. et al. (2019) Emergent tool use from multi-agent interaction. arXiv preprint arXiv:1909.07528.
- Coding With Russ (2024) brawler_tut. GitHub. Available at: https://github.com/russs123/brawler_tut
 [Accessed 12 Nov 2025].
- Hasselt, H.V., Guez, A. and Silver, D. (2016) Deep reinforcement learning with double Q-learning. Proceedings of the AAAI Conference on Artificial Intelligence, 30(1), pp. 2094–2100.
- Mnih, V. et al. (2015) Human-level control through deep reinforcement learning. Nature, 518(7540), pp. 529–533.
- Ng, A.Y., Harada, D. and Russell, S. (1999) Policy invariance under reward transformations: Theory and application to reward shaping. Proceedings of ICML, pp. 278–287.
- Silver, D. et al. (2017) Mastering the game of Go without human knowledge. Nature, 550(7676), pp. 354–359.
- Sutton, R.S. and Barto, A.G. (2018) Reinforcement Learning: An Introduction. 2nd ed. MIT Press.