import argparse
import os
import json
import hashlib
from pathlib import Path

from datasets import load_dataset, load_from_disk
from trl import SFTConfig, SFTTrainer  # type: ignore

import verifiers as vf
import torch
import asyncio
import weave
from transformers import pipeline
from verifiers.inference.vllm_client import VLLMClient
from verifiers.trainers.sft_regularized_sft_trainer import SFTRegularizedSFTTrainer

"""
accelerate launch --config-file configs/zero3.yaml --num-processes 8 examples/sft.py
"""

def make_gen_model_completions(client, model_name: str, temperature: float, top_p: float, max_tokens: int):
    async def _async_generate(messages_batch):
        tasks = [
            client.chat.completions.create(
                model=model_name,
                messages=messages,  # chat-format already
                temperature=temperature,
                top_p=top_p,
                max_tokens=max_tokens,
            )
            for messages in messages_batch
        ]
        responses = await asyncio.gather(*tasks)
        contents = []
        for resp in responses:
            try:
                contents.append(resp.choices[0].message.content)
            except Exception:
                contents.append("")
        return contents

    def _gen_model_completions(batch):
        completions = asyncio.run(_async_generate(batch["prompt"]))
        return {
            "model_completion": [
                [{"role": "assistant", "content": text if isinstance(text, str) else str(text)}]
                for text in completions
            ]
        }

    return _gen_model_completions


def main(args):
    # convenience function for FA2 initialization
    aux_loss_coef = args.aux_loss_coef
    model, tokenizer = vf.get_model_and_tokenizer(args.model, use_liger=False)
    dataset = load_dataset(args.dataset, split="train")

    model.compile()

    # Add model completions as a new column using vLLM if enabled; otherwise fall back to local pipeline
    if args.use_vllm:
        client = VLLMClient(host=getattr(args, "vllm_host", "0.0.0.0"), port=getattr(args, "vllm_port", 8000))
    else:
        client = None

    # Compute a cache key based on dataset fingerprint, model, and generation params
    dataset_fingerprint = getattr(dataset, "_fingerprint", None) or "no_fingerprint"
    cache_payload = {
        "dataset": args.dataset,
        "dataset_fingerprint": dataset_fingerprint,
        "model": args.model,
        "temperature": args.temperature,
        "top_p": args.top_p,
        "max_tokens": args.max_tokens,
        "script_version": "v3",
    }
    cache_key = hashlib.sha256(
        json.dumps(cache_payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    cache_dir = Path(args.mc_cache_dir or os.path.join(args.output_dir, "mc_cache"))
    cache_path = cache_dir / cache_key
    os.makedirs(cache_dir, exist_ok=True)

    if cache_path.exists():
        dataset = load_from_disk(str(cache_path))
    else:
        dataset = dataset.map(
            make_gen_model_completions(
                client,
                args.model,
                temperature=args.temperature,
                top_p=args.top_p,
                max_tokens=args.max_tokens,
            ),
            batched=True,
            batch_size=1024,
        )
        dataset.save_to_disk(str(cache_path))

    tok_counts = []
    for row in dataset:
        # count tokens in (prompt, completion)
        messages = row["prompt"] + row["completion"]  # type: ignore
        toks = tokenizer.apply_chat_template(messages, tokenize=True)
        tok_counts.append(len(toks))

    # tok count stats
    print(f"Dataset size: {len(tok_counts)}")
    print(f"Min tokens: {min(tok_counts)}")
    print(f"Max tokens: {max(tok_counts)}")
    print(f"Mean tokens: {sum(tok_counts) / len(tok_counts)}")
    print(f"Median tokens: {sorted(tok_counts)[len(tok_counts) // 2]}")

    args = SFTConfig(
        max_length=None, # args.max_length,
        output_dir=args.output_dir,
        per_device_train_batch_size=args.per_device_train_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant":False},
        bf16=True,
        learning_rate=args.learning_rate,
        num_train_epochs=args.num_train_epochs,
        weight_decay=args.weight_decay,
        max_grad_norm=args.max_grad_norm,
        report_to="wandb",
        save_strategy="epoch",
        save_total_limit=1,
        logging_steps=10,
        save_only_model=True,
        log_on_each_node=True,
        push_to_hub=args.push_to_hub,
        hub_model_id=args.name_to_save,
    )

    trainer = SFTRegularizedSFTTrainer(
        model=model,
        args=args,
        train_dataset=dataset,  # type: ignore
        aux_loss_coef=aux_loss_coef,
    )
    trainer.train()
    print('done')


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", "-m", type=str, default="Qwen/Qwen3-1.7B-Base")
    parser.add_argument("--dataset", "-d", type=str, default="atrost/math_sft_40K_trl")
    parser.add_argument("--output-dir", "-o", type=str, default="outputs")
    parser.add_argument("--name-to-save", "-n", type=str, default="atrost/math_sft_40K_trl_SFT_Regularized-0.1")
    parser.add_argument("--max-length", "-l", type=int, default=8192)
    parser.add_argument("--per-device-train-batch-size", "-b", type=int, default=1)
    parser.add_argument("--gradient-accumulation-steps", "-a", type=int, default=8)
    parser.add_argument("--learning-rate", "-r", type=float, default=2e-5)
    parser.add_argument("--num-train-epochs", "-e", type=int, default=1)
    parser.add_argument("--weight-decay", "-w", type=float, default=0.01)
    parser.add_argument("--max-grad-norm", "-g", type=float, default=0.1)
    parser.add_argument("--push-to-hub", "-p", type=bool, default=True)
    parser.add_argument("--use-vllm", type=bool, default=False)
    parser.add_argument("--vllm-host", type=str, default="0.0.0.0")
    parser.add_argument("--vllm-port", type=int, default=8000)
    parser.add_argument("--aux-loss-coef", type=float, default=0.1)
    # Sampling params for model completion generation (affect cache key)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--top-p", type=float, default=1.0)
    parser.add_argument("--max-tokens", type=int, default=4096)
    # Cache directory for model completion datasets
    parser.add_argument("--mc-cache-dir", type=str, default=None)
    args = parser.parse_args()
    main(args)
