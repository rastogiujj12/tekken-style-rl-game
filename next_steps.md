⚙️ After This Run Ends — Phase 2 Post-Run Tasks
🧩 Model & Training

Save final weights of both players (policy_net, target_net) with episode tag.

Export optimizer states if resuming later (torch.save(optimizer.state_dict())).

Record final epsilon for each agent.

Back up replay buffers (optional but useful for analysis).

Save RNG seeds if reproducibility matters.

📊 Evaluation & Visualization

Plot:

score_p1 vs episode

score_p2 vs episode

abs(score_p1 – score_p2) vs episode (balance metric)

episode_step vs episode (fight duration trend)

epsilon vs episode (exploration curve)

Compute rolling averages (e.g. window = 20 episodes) for clarity.

Inspect reward distributions (mean ± std) to confirm stability.

Check correlation between score_diff and opponent_ep (should stay ≈ 0).

🧠 Behavioral Testing

Load trained weights and run evaluation matches (no training, rendering on) to visually confirm behavior.

Optionally run fixed-seed matches to compare consistency across episodes.

Record short clips or GIFs for qualitative assessment.

🔍 Further Analysis

Inspect Q-value trends: do they plateau or still rise?

Check loss curves for both agents (converged vs oscillating).

Look for any residual negative-reward dominance.

🪄 Next Experiments

Try loading historical checkpoints (e.g. every 100 episodes) to study skill evolution.

Optionally retrain from best checkpoint with:

Slightly lower LR (e.g. 5e-5)

Smaller ε floor

Opponent sampling from multiple saved policies (for diversity)

Consider fine-tuning in render-off evaluation mode for final scoring stability.