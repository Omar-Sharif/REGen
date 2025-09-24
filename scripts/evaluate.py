#!/usr/bin/env python3
import os
import sys
import json
import argparse
from pathlib import Path

def main():
    parser = argparse.ArgumentParser(description='Evaluate predictions (Exact/Relaxed with optional Complex/JAM)')
    parser.add_argument('--base-path', required=True)
    parser.add_argument('--dataset', required=True)
    parser.add_argument('--model-name', required=True)
    parser.add_argument('--prompt-type', required=True, choices=['zero-shot', 'cot'])
    parser.add_argument('--version', type=int, default=0)
    parser.add_argument('--threshold', type=float, default=0.85)
    parser.add_argument('--do-complex', type=str, default='false', help='true/false')
    parser.add_argument('--judge-mode', choices=['openai', 'hug_api', 'anthropic'], default='openai')
    parser.add_argument('--judge-model-access-string', default='gpt-4o-mini')
    parser.add_argument('--judge-model-name', default='GPT4o')

    args = parser.parse_args()

    base_path = os.path.abspath(args.base_path)

    try:
        from regen.evaluate import (
        get_predictions_for_results,
        getting_role_wise_scores,
        getting_exact_relaxed_match_predictions_dictionary,
        doing_complex_matching,
        getting_complex_match_predictions_dictionary,
    )
    except Exception:
        from Generic_Functions_Evaluation import (
        get_predictions_for_results,
        getting_role_wise_scores,
        getting_exact_relaxed_match_predictions_dictionary,
        doing_complex_matching,
        getting_complex_match_predictions_dictionary,
    )

    # Load processed predictions
    processed_path = os.path.join(base_path, 'Result', args.dataset, f'{args.model_name}-{args.prompt_type}-processed-predictions-{args.dataset}-v{args.version}.json')
    if not os.path.exists(processed_path):
        raise FileNotFoundError(f'Missing processed file: {processed_path}')
    with open(processed_path, 'r') as f:
        processed = json.load(f)

    # Prepare exact/relaxed dictionaries
    er_dict = getting_exact_relaxed_match_predictions_dictionary(
        path=base_path,
        model_name=args.model_name,
        prompt_type=args.prompt_type,
        dataset_name=args.dataset,
        version=args.version,
    )

    unique_roles = list({d['role'] for d in er_dict if 'role' in d})

    # Optionally run complex match via judge model
    do_complex = args.do_complex.lower() == 'true'
    final_dict = er_dict
    judge_name = args.judge_model_name
    if do_complex:
        final_dict = getting_complex_match_predictions_dictionary(
            predictions=er_dict,
            unique_roles=unique_roles,
            judge_model_access_string=args.judge_model_access_string,
            judge_model_name=args.judge_model_name,
            model_origin=args.judge_mode,
            threshold=args.threshold,
        )

    out_dir = os.path.join(base_path, 'Result', args.dataset)
    Path(out_dir).mkdir(parents=True, exist_ok=True)

    # Save after-exact-relaxed file for the requested threshold
    er_out = os.path.join(out_dir, f'{args.model_name}-{args.prompt_type}-after-exact-relaxed-match-predictions-{args.dataset}-v{args.version}.json')
    with open(er_out, 'w') as f:
        json.dump(er_dict, f, ensure_ascii=False, indent=2)

    # Save complex (if any)
    if do_complex:
        cm_out = os.path.join(out_dir, f'{args.model_name}-{args.prompt_type}-after-complex-match-predictions-{args.dataset}-v{args.version}.json')
        with open(cm_out, 'w') as f:
            json.dump(final_dict, f, ensure_ascii=False, indent=2)

    # Load deviation scores
    jds_path = os.path.join(base_path, 'judgment-deviation-scores.json')
    if not os.path.exists(jds_path):
        raise FileNotFoundError(f'Missing deviation file: {jds_path}')
    with open(jds_path, 'r') as f:
        jds = json.load(f)

    # Compute role-wise metrics
    role_scores = getting_role_wise_scores(
        predictions=final_dict,
        unique_roles=unique_roles,
        threshold=args.threshold,
        judge_model=judge_name,
        dataset_name=args.dataset,
        judgement_deviation_score=jds,
    )

    scores_out = os.path.join(out_dir, f'{args.model_name}-{args.prompt_type}-scores-{args.dataset}-v{args.version}.json')
    with open(scores_out, 'w') as f:
        json.dump(role_scores, f, ensure_ascii=False, indent=2)

    print(f'Saved exact/relaxed dict: {er_out}')
    if do_complex:
        print(f'Saved complex dict: {cm_out}')
    print(f'Saved scores: {scores_out}')


if __name__ == '__main__':
    main()
