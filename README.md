# Battle Agents

A self‑play, deep‑reinforcement‑learning (DQN)–powered 2D fighting simulator built with Pygame and PyTorch. In **Battle Agents**, two fighters learn to battle each other over thousands of rounds—discovering strategies, combos, and even emergent tricks like double‑jumps.

---

## 🚀 Features

- **Deep Q‑Networks (DQN):** Each agent has its own 5‑layer neural network (4 inputs → 256, 256, 128, 64 hidden units → 5 outputs).
- **Self‑play training:** Fighters learn by competing against each other, driving an “arms race” of tactics.
- **Emergent behavior:** Watch them invent hit‑and‑run, corner‑trap pressure, timed heavy attacks, and even double‑jumps!
- **Replay buffer & target network:** Stable training with experience replay and periodic target‑network updates.
- **Extensible architecture:** Easily add new actions (dash, block, combos), tweak rewards, or change arena settings.

---

## 📂 Repository Structure

```
Battle-Agents/
├── assets/
│   ├── images/
│   │   ├── background/
│   │   └── warrior/, wizard/ sprites
│   └── audio/
│       ├── music.mp3
│       └── sword.wav, magic.wav
├── main.py          # Game loop, window setup, score & round logic
├── fighter.py       # DQN implementation, agent physics & actions
├── requirements.txt # Python dependencies (pygame, torch, numpy)
└── README.md        # You are here!
```

---

## 🛠 Getting Started

1. **Clone the repo**
   ```bash
   git clone https://github.com/buzzfit/Battle-Agents.git
   cd Battle-Agents
   ```

2. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

3. **Run the simulator**
   ```bash
   python main.py
   ```

4. **Watch the learning** — agents train continuously. After hundreds or thousands of rounds, they evolve sophisticated tactics that you can observe in real time.

---

---

*This simulator is fully autonomous: the agents train and play against each other without manual input.*

---

## ⚙ Technical Details

- **State vector:** `[dx_norm, dy_norm, self_health, opponent_health]` (4 dims)
- **Action space:** 5 discrete actions (idle/move left/move right/jump/light/heavy attack)
- **Reward structure:**
  - +1.0 for a successful hit
  - –0.1 for a missed attack
  - –5.0 on death
- **Network & training:** Adam optimizer (lr=1e-4), γ=0.99, ε-decay from 1.0 → 0.1, replay buffer size=10k, batch=64, target update every 1k steps.

---

## 📈 Results & Observations

- **Balanced play:** After 400 rounds, agents often split wins 200/200—a near‑perfect draw.
- **Emergent moves:** Learned combos, bait‑and‑punish sequences, and double‑jumps without explicit coding.

---

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/<name>`)
3. Commit your changes (`git commit -m "Add <feature>"`)
4. Push to your branch (`git push origin feature/<name>`)
5. Open a Pull Request

We welcome ideas: new moves, arena shapes, multi‑agent tournaments, visualizations, and more!

---

## 🙏 Acknowledgements

This project builds upon the original manual player-vs-player tutorial by Russs123. Thanks to [Brawler Tutorial on GitHub](https://github.com/russs123/brawler_tut) and the [YouTube walkthrough video](https://www.youtube.com/watch?v=s5bd9KMSSW4) for the inspiration and foundational code.

---

﻿# Battle-Agents
