"""Launch one controlled ablation; extra arguments are passed to train.py."""
import argparse
import json
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parent


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--experiment', required=True)
    parser.add_argument('--config', type=Path, default=ROOT / 'ablations.json')
    parser.add_argument('--dry-run', action='store_true')
    args, training_args = parser.parse_known_args()
    experiments = json.loads(args.config.read_text(encoding='utf-8'))
    if args.experiment not in experiments:
        parser.error(f"Unknown experiment; choose from {list(experiments)}")
    experiment = experiments[args.experiment]
    # Explicit settings pin the architecture/control family; later args intentionally override.
    command = [sys.executable, str(ROOT / 'train.py'), '--model', 'gnn_lstm',
               '--gnn-layers', '2', '--connectivity', '8', '--aggregation', 'mean',
               '--temporal-pool', 'last', '--objective', 'graph_batch',
               '--output-dir', f'runs/{args.experiment}', *experiment['args'], *training_args]
    print(experiment['description'], flush=True)
    print(subprocess.list2cmdline(command), flush=True)
    if not args.dry_run:
        subprocess.run(command, check=True)


if __name__ == '__main__':
    main()
