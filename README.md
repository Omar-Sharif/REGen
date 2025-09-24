## REGen: A Reliable Evaluation Framework for Generative Event Argument Extraction
[Paper link](https://arxiv.org/pdf/2502.16838) (Accepted in EMNLP-2025 Findings)

This repo folder provides reusable scripts to run prediction, post-processing, and evaluation for event argument extraction across multiple datasets for our REGen framework:
- DiscourseEE, DocEE, RAMS, WikiEvents, PHEE, GENEVA, DICE

### Prerequisites
- Python 3.10+
- API access for using LLMs
  - OpenAI: set `OPENAI_API_KEY`
  - HuggingFace Inference API: set `HUGGINGFACEHUB_API_TOKEN`

### Install
Create and activate a conda environment named REGen:
```bash
conda create -n REGen python=3.10 -y
conda activate REGen
pip install -r requirements.txt
```
### Data Preprocessing
All the datasets are preprocess following the description in Section 2.1 and Appendix C in our paper. Notably we took an trigger-free approach to prepare the dataset. 

### Data and Outputs
- **Datasets**: `Data/<DatasetName>/` (e.g., `Data/RAMS/`)
- **Outputs**: `Result/<DatasetName>/` (e.g., `Result/RAMS/`)

### Datasets Expected Format
Each dataset directory contains:
- `<DatasetName>_schema.json`: event schema required by prompts and evaluation
- `*-processed-*-data-*.json` or `*.jsonl`: preprocessed inputs for prediction

Example (RAMS):
- `Data/RAMS/RAMS_schema.json`
- `Data/RAMS/RAMS-processed-test-data-EAE-Eval.json`

### Environment Variables
Set keys only for the providers you use. We have used huggingface endpoints for our experiments
```bash
export OPENAI_API_KEY=your_openai_key            # for --mode openai
export HUGGINGFACEHUB_API_TOKEN=your_hf_token    # for --mode hug_api
```

### Command Line Overview
Common arguments:
- `--dataset`: one of RAMS, DocEE, DiscourseEE, WikiEvents, PHEE, GENEVA
- `--model-name`: short tag used in filenames (e.g., `GPT4o`, `Llama3.1-70B`, `Phi-3.5`, `Gemma1.1-7B`, `Mixtral-8x7B`)
- `--prompt-type`: `zero-shot` or `cot`
- `--version`: integer tag added to filenames (default: 0)

LLM-Inference:
- `--mode`: `openai`, `hug_api`, or `anthropic` (we did not run experiment with anthropic
- `--model-access-string`: provider model id (e.g., `gpt-4o-2024-11-20`, `meta-llama/Llama-3.1-70B-Instruct`, `mistralai/Mixtral-8x7B-Instruct-v0.1`, `google/gemma-1.1-7b-it`, `microsoft/Phi-3.5-mini-instruct`). Different model string we have used for openai and hugginface models

Evaluation:
- `--threshold`: relaxed/complex threshold (default 0.85). We define this threshold through human judgement. Please check the paper.
- `--do-complex`: `true`/`false` to run complex matching via an LLM judge
- `--judge-mode`, `--judge-model-access-string`, `--judge-model-name`: judge model config when `--do-complex true`

### Quickstart: End-to-End (RAMS Example)
From the project root:

1) Predict
```bash
python scripts/predict.py \
  --base-path "$(pwd)" \
  --dataset RAMS \
  --mode openai \
  --model-access-string gpt-4o-2024-11-20 \
  --model-name GPT-4o \
  --prompt-type zero-shot \
  --version 0
```
Creates: `Result/RAMS/GPT4o-zero-shot-raw-predictions-RAMS-v0.json`

2) Process raw predictions
```bash
python scripts/process_predictions.py \
  --base-path "$(pwd)" \
  --dataset RAMS \
  --model-name GPT-4o \
  --prompt-type zero-shot \
  --version 0
```
Creates: `Result/RAMS/GPT4o-zero-shot-processed-predictions-RAMS-v0.json`

3) Evaluate (exact/relaxed)
```bash
python scripts/evaluate.py \
  --base-path "$(pwd)" \
  --dataset RAMS \
  --model-name GPT-4o \
  --prompt-type zero-shot \
  --version 0 \
  --threshold 0.85 \
  --do-complex false
```
Creates:
- `...-after-exact-relaxed-match-predictions-...json`
- `...-scores-...json`
(The code prints all the matching arguments when this script is running. )

4) Optional: Complex matching + JAM
```bash
python scripts/evaluate.py \
  --base-path "$(pwd)" \
  --dataset RAMS \
  --model-name GPT-4o \
  --prompt-type zero-shot \
  --version 0 \
  --threshold 0.85 \
  --do-complex true \
  --judge-mode openai \
  --judge-model-access-string gpt-4o-2024-11-20 \
  --judge-model-name GPT4o
```
Adds: `...-after-complex-match-predictions-...json`

### Usage 
- Feel free to use your own judge model and change code accordingly.
- The evaluation is setup in a way this works best for when you have exact, relaxed, and complex match predictions. If complex match predictions are missing you might need to changes the function accordingly.


### Directory Structure
```
REGen/
├── Data/                    # Input datasets
│   ├── RAMS/
│   │   ├── RAMS_schema.json
│   │   └── RAMS-processed-test-data-EAE-Eval.json
│   └── ...
├── Result/                  # Output files
│   ├── RAMS/
│   │   └── <generated files>
│   └── ...
├── scripts/                 # Command-line tools
│   ├── predict.py
│   ├── process_predictions.py
│   └── evaluate.py
├── src/regen/              # Python package
│   ├── __init__.py
│   ├── predictions.py
│   ├── process.py
│   └── evaluate.py
├── requirements.txt        # Dependencies
├── README.md              # This file
├── Generic-Statistics-Results.ipynb    # Notebook for general statistics and evaluation metrics
├── Head-Noun-Phrase-Matching.ipynb     # Notebook for head noun phrase matching analysis
└── judgment-deviation-scores.json      # For JAM evaluation
```

### Notes
-  All code, data, and outputs are organized in the root directory
-  Please use appropriate API keys
-  `judgment-deviation-scores.json` is used for JAM evaluation
-  Core logic is in the `src/regen/` package, scripts provide CLI interface

### Analysis Notebooks

Check two notebooks to get additional analysis on top of the core pipeline. They assume the project is opened at the repository root (they dynamically use `pwd` for paths).

- **General statistics** (`Generic-Statistics-Results.ipynb`)
  - Computes dataset-level stats (instances, arguments, avg doc/argument length, density) and aggregates evaluation metrics across models/prompt types (EM/RM/CM/JAM). Also visualizes inference-count reduction from REGen vs LLM-as-Judge.
  - Inputs: Uses files under `Result/<Dataset>/` produced by the main pipeline and schemas under `Data/<Dataset>/`.
  - Outputs: Prints tables in figures in the notebook s (e.g., `inference-count-and-reduction-comparison.png`, `performance-comparison.png`) 

- **Head noun phrase matching** (`Head-Noun-Phrase-Matching.ipynb`)
  - Provides comparison results with Head-Noun-Match-Phrase approach and our REGen approach
  - Here we run experiments to check when the head nouns of predictions and gold annotations match. Reports HM precision/recall/F1 across datasets/models/prompt types and can save per-dataset HM-adjusted prediction files to `Result/<Dataset>/` with the pattern: `{model}-{prompt}-after-HM-predictions-{dataset}-v{version}.json`.
  - Requirements: `spacy` with the small English model installed. If missing, install and download via:
    - `pip install spacy`
    - `python -m spacy download en_core_web_sm`
  - Inputs: Uses processed predictions from `Result/<Dataset>/` and the corresponding schema in `Data/<Dataset>/`.
  - Outputs: HM metrics in the notebook and optional HM-adjusted predictions under `Result/<Dataset>/`.


### Potential Errors
- **FileNotFoundError**: Ensure dataset files like `RAMS-processed-test-data-EAE-Eval.json` exist in `Data/<DatasetName>/`
- **API errors**: Scripts retry automatically, but you may need to slow down or switch providers
- **Missing API keys**: Export `OPENAI_API_KEY` or `HUGGINGFACEHUB_API_TOKEN` environment variables
- **Import errors**: Make sure you're in the project root and have installed dependencies


### Citation

If you use this codebase or results in your research, please cite:

```bash
@misc{sharif2025regenreliableevaluationframework,
      title={REGen: A Reliable Evaluation Framework for Generative Event Argument Extraction}, 
      author={Omar Sharif and Joseph Gatto and Madhusudan Basak and Sarah M. Preum},
      year={2025},
      eprint={2502.16838},
      archivePrefix={arXiv},
      primaryClass={cs.CL},
      url={https://arxiv.org/abs/2502.16838}, 
}
```




