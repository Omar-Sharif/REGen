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
    parser = argparse.ArgumentParser(description='Run LLM predictions and save raw outputs')
    parser.add_argument('--base-path', required=True, help='Path to REGen project root (contains Data/ and Result/)')
    parser.add_argument('--dataset', required=True, help='Dataset name, e.g., RAMS, DocEE, DiscourseEE, WikiEvents, PHEE, GENEVA, DICE')
    parser.add_argument('--mode', required=True, choices=['openai', 'hug_api', 'anthropic'])
    parser.add_argument('--model-access-string', required=True, help='Provider-specific model identifier')
    parser.add_argument('--model-name', required=True, help='Short model tag for filenames (e.g., GPT4o)')
    parser.add_argument('--prompt-type', required=True, choices=['zero-shot', 'cot'])
    parser.add_argument('--version', type=int, default=0)

    args = parser.parse_args()

    base_path = os.path.abspath(args.base_path)
    ensure_parent_on_path(base_path)

    from regen.predictions import get_test_data_and_schema, get_model_predictions

    data_dir = os.path.join(base_path, 'Data')
    data, event_schema = get_test_data_and_schema(data_dir, args.dataset)

    raw_predictions, inference_cnt = get_model_predictions(
        data=data,
        dataset_name=args.dataset,
        event_schema=event_schema,
        mode=args.mode,
        model_name=args.model_name,
        model_access_string=args.model_access_string,
        prompt_type=args.prompt_type,
    )

    out_dir = os.path.join(base_path, 'Result', args.dataset)
    Path(out_dir).mkdir(parents=True, exist_ok=True)
    out_file = os.path.join(out_dir, f'{args.model_name}-{args.prompt_type}-raw-predictions-{args.dataset}-v{args.version}.json')

    with open(out_file, 'w') as f:
        json.dump(raw_predictions, f, ensure_ascii=False, indent=2)

    print(f'Saved raw predictions: {out_file}')
    print(f'Inference count: {inference_cnt}')


if __name__ == '__main__':
    main()
