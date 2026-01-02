#!/usr/bin/env bash
set -euo pipefail

# Fill in your API tokens before running.
export OPENAI_API_KEY="YOUR_OPENAI_KEY"
export GEMINI_API_KEY="YOUR_GEMINI_KEY"
export OPENROUTER_API_KEY="YOUR_OPENROUTER_KEY"
export HUGGINGFACE_API_KEY="YOUR_HF_KEY"

# Fixed settings; edit here as needed.
MODEL="gpt-3.5-turbo"
REPEATS=1
MAX_QUESTIONS=50
CONCURRENCY=5   # start ~5; adjust based on rate limits.
TEMP=1.0
TOP_P=1.0
MAX_TOKENS=256
OUTPUT_DIR="outputs"

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

python "$ROOT_DIR/scripts/generate_response.py" \
  --model "$MODEL" \
  --temperature "$TEMP" \
  --top-p "$TOP_P" \
  --max-tokens "$MAX_TOKENS" \
  --repeats "$REPEATS" \
  --max-questions "$MAX_QUESTIONS" \
  --concurrency "$CONCURRENCY" \
  --output-dir "$OUTPUT_DIR"

