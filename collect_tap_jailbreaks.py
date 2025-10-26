"""
Collect TAP jailbreaks
"""
import os
import pandas as pd
import re


FOLDER = "tap_qwen3"
collected = {}
for fp in os.listdir(FOLDER):
    if not fp.endswith(".parquet"):
        continue
    path = os.path.join(FOLDER, fp)
    num_iter = int(re.search(r"iter_(\d+)_df\.parquet", fp).group(1))
    # Read file
    df = pd.read_parquet(path)
    # Pick 'closest to end' prompt with best judge score
    last_best = df[df['judge_scores'] == df['judge_scores'].max()].iloc[-1]
    collected[num_iter] = last_best.prompt

# Create file
sorted_prompts = [collected[k] for k in sorted(collected.keys())]
print(f"{len(sorted_prompts)} prompts!")

with open("tap_qwen3.txt", "w") as f:
    for v in sorted_prompts:
        f.write(f"{v}\n")