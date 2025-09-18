import argparse
import os
import json
import hashlib
import shutil
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
    normalize_loss = args.normalize_loss
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

    # create run name
    run_name = f"{args.run_name_base}_Regularized-{aux_loss_coef}_Normalize-{normalize_loss}"

    # create name to save
    name_to_save = f"{args.hf_username}/{run_name}"

    output_dir = f"{args.output_dir}/{run_name}"

    args = SFTConfig(
        max_length=None, # args.max_length,
        output_dir=output_dir,
        per_device_train_batch_size=args.per_device_train_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant":False},
        bf16=True,
        learning_rate=args.learning_rate,
        num_train_epochs=args.num_train_epochs,
        weight_decay=args.weight_decay,
        max_grad_norm=args.max_grad_norm,
        lr_scheduler_type=args.lr_scheduler_type,
        warmup_ratio=args.warmup_ratio,
        report_to="wandb",
        save_strategy="epoch",
        save_total_limit=1,
        logging_steps=1,
        save_only_model=True,
        log_on_each_node=True,
        push_to_hub=args.push_to_hub,
        hub_model_id=name_to_save,
        run_name=run_name,
    )

    trainer = SFTRegularizedSFTTrainer(
        model=model,
        args=args,
        train_dataset=dataset,  # type: ignore
        aux_loss_coef=aux_loss_coef,
        normalize_loss=normalize_loss,
    )
    trainer.train()

    # delete output dir
    print(f"Deleting directory '{output_dir}'")
    if os.path.exists(output_dir):
        try:
            shutil.rmtree(output_dir)
            print(f"Directory '{output_dir}' and its contents deleted successfully.")
        except OSError as e:
            print(f"Error: {output_dir} : {e.strerror}")
    else:
        print(f"Directory '{output_dir}' does not exist.")

    print('done')



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="Qwen/Qwen3-1.7B-Base")
    parser.add_argument("--dataset", type=str, default="atrost/math_sft_40K_trl")
    parser.add_argument("--output-dir", type=str, default="outputs")
    parser.add_argument("--hf-username", type=str, default="atrost")
    parser.add_argument("--run-name-base", type=str, default="math_sft_40K_trl_SFT")
    parser.add_argument("--max-length", type=int, default=8192)
    parser.add_argument("--per-device-train-batch-size", type=int, default=1)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=64)
    parser.add_argument("--learning-rate", type=float, default=5e-5)
    parser.add_argument("--num-train-epochs", type=int, default=1)
    parser.add_argument("--weight-decay", type=float, default=0.00)
    parser.add_argument("--max-grad-norm", type=float, default=1.0)
    parser.add_argument("--lr-scheduler-type", type=str, default="cosine")
    parser.add_argument("--warmup-ratio", type=float, default=0.1)
    parser.add_argument("--push-to-hub", dest="push_to_hub", action="store_true", help="Enable pushing to the Hub")
    parser.add_argument("--use-vllm", dest="use_vllm", action="store_true", help="Enable vLLM client for model completions")
    parser.add_argument("--vllm-host", type=str, default="127.0.0.1")
    parser.add_argument("--vllm-port", type=int, default=8000)
    parser.add_argument("--aux-loss-coef", type=float, default=0.1)
    parser.add_argument("--normalize-loss", dest="normalize_loss", action="store_true", help="Normalize combined loss by (1 + aux_coef)")
    # Sampling params for model completion generation (affect cache key)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--top-p", type=float, default=1.0)
    parser.add_argument("--max-tokens", type=int, default=4096)
    # Cache directory for model completion datasets
    parser.add_argument("--mc-cache-dir", type=str, default=None)
    args = parser.parse_args()
    main(args)
