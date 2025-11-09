import pandas as pd

for f in [
    "logs/phase_1_episodes_20251107_125500.csv",
    "logs/phase_1_rewards_20251107_125500.csv",
    "logs/phase_1_steps_20251107_125500.csv",
]:
    df = pd.read_csv(f)
    print(f"\n{f} — shape {df.shape}")
    print(df.head())