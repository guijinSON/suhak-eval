import argparse
import asyncio
import os
import re
import pandas as pd
from datasets import load_dataset
import litellm
from litellm import acompletion
from tqdm.asyncio import tqdm_asyncio



def parse_args():
    parser = argparse.ArgumentParser(description="Query questions with LiteLLM.")
    parser.add_argument("--model", required=True, help="Model name to call.")
    parser.add_argument(
        "--system",
        type=str,
        default=None,
        help="Optional system prompt to prepend.",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=1.0,
        help="Sampling temperature.",
    )
    parser.add_argument(
        "--top-p",
        dest="top_p",
        type=float,
        default=None,
        help="Nucleus sampling probability mass.",
    )
    parser.add_argument(
        "--max-tokens",
        dest="max_tokens",
        type=int,
        default=None,
        help="Max tokens to generate.",
    )
    parser.add_argument(
        "--repeats",
        type=int,
        default=1,
        help="How many times to query each question.",
    )
    parser.add_argument(
        "--max-questions",
        type=int,
        default=None,
        help="Optional cap on number of questions to process.",
    )
    parser.add_argument(
        "--concurrency",
        type=int,
        default=5,
        help="Maximum concurrent requests.",
    )
    parser.add_argument(
        "--output-dir",
        default="outputs",
        help="Directory to save results; filename derived from model.",
    )
    return parser.parse_args()


def extract_text(resp):
    try:
        return resp.choices[0].message["content"]
    except Exception:
        return ""


async def fetch_answer(model, system_prompt, question, row_data, q_idx, run_idx, sem, temperature, top_p, max_tokens):
    async with sem:
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": question})

        kwargs = {
            "model": model,
            "messages": messages,
        }
        if temperature is not None:
            kwargs["temperature"] = temperature
        if top_p is not None:
            kwargs["top_p"] = top_p
        if max_tokens is not None:
            kwargs["max_tokens"] = max_tokens

        resp = await acompletion(**kwargs)
    text = extract_text(resp)

    return {
        **row_data,
        "question_index": q_idx,
        "run_index": run_idx,
        "generation": text,
        "model": model,
        "temperature": temperature,
        "top_p": top_p,
        "max_tokens": max_tokens,
    }


async def main():
    args = parse_args()
    sem = asyncio.Semaphore(max(1, args.concurrency))

    df = load_dataset("amphora/suhak-full")["train"].to_pandas()
    df = df[df["question"].notna()].reset_index(drop=True)
    if args.max_questions:
        df = df.head(args.max_questions)

    os.makedirs(args.output_dir, exist_ok=True)
    sanitized_model = re.sub(r"[^A-Za-z0-9_.-]+", "_", args.model)
    out_path = os.path.join(args.output_dir, f"{sanitized_model}_generations.csv")

    tasks = []
    for q_idx, row in df.iterrows():
        question = row["question"]
        row_data = row.to_dict()
        for run_idx in range(1, args.repeats + 1):
            tasks.append(
                fetch_answer(
                    args.model,
                    args.system,
                    question,
                    row_data,
                    q_idx,
                    run_idx,
                    sem,
                    args.temperature,
                    args.top_p,
                    args.max_tokens,
                )
            )

    # Create tasks so we can consume completions with progress.
    tasks = [asyncio.create_task(coro) for coro in tasks]
    results = []
    for fut in tqdm_asyncio.as_completed(tasks, total=len(tasks)):
        result = await fut
        results.append(result)
    out_df = pd.DataFrame(results)
    out_df.to_csv(out_path, index=False)
    print(f"Saved {len(out_df)} rows to {out_path}")


if __name__ == "__main__":
    asyncio.run(main())
