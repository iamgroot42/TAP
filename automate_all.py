import os 
import json
import os
import subprocess
from pathlib import Path
from typing import List
from tqdm import tqdm


def read_prompts(filepath: Path) -> List[dict]:
    with open(filepath, 'r') as f:
        data = json.load(f)
    samples = data['samples']
    collected = []
    for m in samples:
        collected.append({
            'id': m['id'],
            'goal': m['prompt'],
            'target': m['target'],
        })
    return collected


def run_simulation(data: List[dict], target: str):

    folder = f'tap_{target}/'
    # Make sure above directory exists
    os.makedirs(folder, exist_ok=True)

    for index, record in enumerate(tqdm(data), start=1):
        # Skip index if file already exists
        if os.path.exists(Path(folder) / f'iter_{index}_df.parquet'):
            continue

        goal_str = record['goal']
        target_str = record['target']

        command = [
            "python",
            "main_TAP.py",
            "--target-model", target,
            "--goal", goal_str,
            "--target-str", target_str,
            "--store-folder", folder,
            "--iter-index", str(index),
        ]

        log_path = Path(folder) / f"iter_{index}"

        # Capture stdout/stderr in the same log file to match the previous shell redirection
        with open(log_path, "a", encoding="utf-8") as log_file:
            subprocess.run(command, stdout=log_file, stderr=subprocess.STDOUT, check=False)


def main():
    prompts_data = read_prompts(PROMPTS_PATH)

    run_simulation(
        data=prompts_data,
        target=TARGET_MODEL 
    )


if __name__ == "__main__":
    PROMPTS_PATH="jailbreak_oracle_benchmark.json"
    TARGET_MODEL="qwen3"

    main()
