#!/usr/bin/env

# Fill in your API tokens before running.
# export OPENAI_API_KEY="YOUR_OPENAI_KEY"
export GEMINI_API_KEY=
export OPENROUTER_API_KEY=
export HF_TOKEN=

# Fixed settings; edit here as needed.
MODEL="gemini/gemini-2.5-flash"
REPEATS=2
MAX_QUESTIONS=10
CONCURRENCY=32   # start ~5; adjust based on rate limits.
TEMP=1.0
TOP_P=""
MAX_TOKENS=8192
OUTPUT_DIR="outputs"
SYSTEM_PROMPT="return your final answer in \\boxed{N} form."

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

python "$ROOT_DIR/scripts/generate_response.py" \
  --model "$MODEL" \
  --temperature "$TEMP" \
  --max-tokens "$MAX_TOKENS" \
  --repeats "$REPEATS" \
  --max-questions "$MAX_QUESTIONS" \
  --concurrency "$CONCURRENCY" \
  --output-dir "$OUTPUT_DIR" \
  ${TOP_P:+--top-p "$TOP_P"} \
  ${SYSTEM_PROMPT:+--system "$SYSTEM_PROMPT"}

