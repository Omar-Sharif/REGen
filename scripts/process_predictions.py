#!/usr/bin/env python3
import os
import sys
import json
import argparse
from pathlib import Path

def ensure_parent_on_path(base_path: str):
    # Add src to path for regen package
    src_dir = os.path.join(base_path, 'src')
    if os.path.isdir(src_dir) and src_dir not in sys.path:
        sys.path.insert(0, src_dir)
    # Fallback to old Code/ if needed
    code_dir = os.path.join(base_path, 'Code')
    if os.path.isdir(code_dir) and code_dir not in sys.path:
        sys.path.append(code_dir)


def main():
    parser = argparse.ArgumentParser(description='Process raw predictions to normalized predictions')
    parser.add_argument('--base-path', required=True, help='Path to REGen project root (contains Data/ and Result/)')
    parser.add_argument('--dataset', required=True)
    parser.add_argument('--model-name', required=True)
    parser.add_argument('--prompt-type', required=True, choices=['zero-shot', 'cot'])
    parser.add_argument('--version', type=int, default=0)

    args = parser.parse_args()

    base_path = os.path.abspath(args.base_path)
    ensure_parent_on_path(base_path)

    from regen.process import get_raw_predictions_and_schema, list_normalization, get_process_predictions

    raw_predictions, event_schema = get_raw_predictions_and_schema(
        path=base_path,
        dataset_name=args.dataset,
        model_name=args.model_name,
        prompt_type=args.prompt_type,
        version=args.version
    )

    processed = []
    for i, dt in enumerate(raw_predictions):
        new_dt = dict(dt)
        role = dt['role']
        raw_pred = dt['raw-initial-predictions']
        values, _ = get_process_predictions(raw_pred, role, i)
        values = [v for v in values if isinstance(v, str)]
        values = list_normalization(values)
        new_dt['initial-predictions'] = values
        processed.append(new_dt)

    out_dir = os.path.join(base_path, 'Result', args.dataset)
    Path(out_dir).mkdir(parents=True, exist_ok=True)
    out_file = os.path.join(out_dir, f'{args.model_name}-{args.prompt_type}-processed-predictions-{args.dataset}-v{args.version}.json')

    with open(out_file, 'w') as f:
        json.dump(processed, f, ensure_ascii=False, indent=2)

    print(f'Saved processed predictions: {out_file}')


if __name__ == '__main__':
    main()
