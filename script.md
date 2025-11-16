✅ 🎥 Final Video Script (8–10 minutes)

“Adaptive Game AI Using Multi-Phase Deep Q-Learning”

🎬 0:00 – 0:20 — Introduction

Hi, my name is Ujjwal Rastogi, and this is my project for Adaptive Game AI using Reinforcement Learning. I built a 2D fighting game NPC that learns progressively through a multi-phase Deep Q-Learning system and adapts its difficulty dynamically. The goal was to make the NPC behave more human-like instead of following predictable, scripted patterns.

🎬 0:20 – 1:20 — What Problem I’m Solving

Most NPCs in games rely on fixed scripts. That makes them easy to memorise, and players eventually find patterns. My objective was to create an NPC that:

learns from experience

improves over time

adapts to different opponents

shows human-like combat behaviours: spacing, timing, engaging, retreating

To achieve this, I implemented a multi-phase DQN training system, each phase focusing on different aspects of the skill tree.

🎬 1:20 – 2:30 — Game Environment Overview

The game is based on a 2D fighting template built in PyGame, modified heavily to support reinforcement learning.

The agent receives a 6-dimensional state vector:

Horizontal distance

Relative x positions

Vertical velocity

Its own health

Opponent health

Attack cooldown

The agent has 5 discrete actions:

Move left

Move right

Jump

Attack 1

Attack 2

During training, the agent interacts with the environment frame-by-frame, collecting rewards for good combat strategy and penalties for bad habits.

🎬 2:30 – 4:00 — Multi-Phase Learning (Important Part)
🔵 Phase 1 — Basic Competence

In Phase 1, the focus is extremely simple: teach the agent the fundamentals.

Reward function for Phase 1 encourages:

moving closer to the opponent

landing hits

not running away

small survival rewards

This gives the agent the foundation — it learns how to move, chase, jump, and attack.

🟢 Phase 2 — Adaptive, Human-Like Behaviour

Phase 2 is where the agent starts becoming strategic.

The reward function encourages:

proper spacing (not too close, not too far)

counter-timing

rewarding hits more than random movement

small “mercy” reward when agent leads in health

penalising one-sided, very short fights

Also in Phase 2, the agent fights snapshots of its own previous versions, giving it diverse opponents.

This creates variation, unpredictability, and basic dynamic difficulty adjustment.

🔴 Phase 3 — Stability & Specialisation

Phase 3 simplifies the reward heavily to stabilise training:

a Gaussian spacing reward centered at ideal fighting distance

small per-frame survival bonus

higher reward for actual hits

softer penalties on misses and jumps

minimal shaping for general movement

This phase helps the agent stop “gaming the reward” and makes behaviours more consistent and cleaner.

🎬 4:00 – 5:30 — DQN Architecture

Show the diagram or code snippet.

A small but efficient neural network:

6-input state

128-unit hidden layer

5-output Q values (one for each action)

I used Double DQN to reduce overestimation bias, experience replay to stabilise updates, and a separate target network for stable training. Exploration is controlled through epsilon-greedy decay across phases.

🎬 5:30 – 7:00 — Results (No plots, just gameplay explanation)

Now let’s look at what the agent learned.

Phase 1 Gameplay

Moves toward opponent

Attempts attacks

Often jumps to dodge

Basic chase-and-hit behaviour

Phase 2 Gameplay

Maintains fighting range

Runs in, attacks, retreats

Shows counter-attacks

Adapts to different snapshots

Sometimes shows “mercy” when opponent is weak

Phase 3 Gameplay

Much smoother spacing

Fewer random jumps

Better hit timing

Higher consistency

Less chaotic movement

The agent isn’t perfect, but exhibits noticeably more human-like behaviours than a scripted bot.

🎬 7:00 – 8:00 — Additional Work: Scripted NPC for Demo

For demonstration purposes, I also created a lightly scripted NPC to show how the RL agent compares.

This scripted NPC:

maintains distance

attacks at the right range

becomes aggressive when losing

becomes passive when winning

performs occasional feints, jumps, or jitters

This lets me show a clean comparison between “handcrafted AI” and “learned AI”.

🎬 8:00 – 9:00 — Challenges

There were challenges:

balancing reward shaping

removing degenerate strategies like constant jumping

stabilising training over thousands of episodes

ensuring opponents are diverse enough

keeping the agent from overfitting to specific snapshots

Still, multi-phase training helped significantly.

🎬 9:00 – 9:45 — Future Work

continuous self-play

population-based training

dueling networks / prioritised replay

larger action space including “idle”

combo actions and move chains

🎬 9:45 – 10:00 — Closing

In summary, this project shows how reinforcement learning can create more dynamic, human-like NPC behavior in fighting games. Multi-phase training proved effective for building skills progressively, and dynamic difficulty emerged naturally through opponent diversity.

Thank you.

If you want, I can also produce:
✔ shorter 5-minute version
✔ with humour
✔ with more technical tone
✔ with more narrative tone
✔ completely polished teleprompter-style script

Just tell me.