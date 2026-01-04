#!/usr/bin/env

# Fill in your API tokens before running.
# export OPENAI_API_KEY="YOUR_OPENAI_KEY"
export GEMINI_API_KEY=
export OPENROUTER_API_KEY=
export HF_TOKEN=

MODELS=(
  "gemini/gemini-2.5-pro"
  "gpt-5.2"
  "gpt-5-mini"
)

REPEATS=3
CONCURRENCY=32   # start ~5; adjust based on rate limits.
TEMP=1.0
TOP_P=""
MAX_TOKENS=32768
OUTPUT_DIR="outputs"
SYSTEM_PROMPT='Solve the provided question. Return your final answer in \\boxed{N} form.'

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

for MODEL in "${MODELS[@]}"; do
  echo "==> Running model: $MODEL"

  # Make a filesystem-safe name for per-model output folders
  MODEL_SAFE="${MODEL//\//__}"
  MODEL_OUTDIR="${OUTPUT_DIR}/${MODEL_SAFE}"

  python "$ROOT_DIR/scripts/generate_response.py" \
    --model "$MODEL" \
    --temperature "$TEMP" \
    --max-tokens "$MAX_TOKENS" \
    --repeats "$REPEATS" \
    --concurrency "$CONCURRENCY" \
    --output-dir "$MODEL_OUTDIR" \
    ${TOP_P:+--top-p "$TOP_P"} \
    ${SYSTEM_PROMPT:+--system "$SYSTEM_PROMPT"}
done