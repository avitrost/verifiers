import argparse

from datasets import load_dataset
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

def make_gen_model_completions(client, model_name: str):
    async def _async_generate(messages_batch):
        tasks = [
            client.chat.completions.create(
                model=model_name,
                messages=messages,  # chat-format already
                temperature=1.0,
                top_p=1.0,
                max_tokens=4096,
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
    model, tokenizer = vf.get_model_and_tokenizer(args.model, use_liger=False)
    dataset = load_dataset(args.dataset, split="train").select(range(16))

    # Add model completions as a new column using vLLM if enabled; otherwise fall back to local pipeline
    client = VLLMClient(host=getattr(args, "vllm_host", "0.0.0.0"), port=getattr(args, "vllm_port", 8000))

    dataset = dataset.map(make_gen_model_completions(client, args.model), batched=True, batch_size=8)

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
        max_length=args.max_length,
        output_dir=args.output_dir,
        per_device_train_batch_size=args.per_device_train_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        gradient_checkpointing=True,
        bf16=True,
        learning_rate=args.learning_rate,
        num_train_epochs=args.num_train_epochs,
        weight_decay=args.weight_decay,
        max_grad_norm=args.max_grad_norm,
        report_to="wandb",
        save_strategy="epoch",
        save_total_limit=1,
        logging_steps=1,
        save_only_model=True,
        log_on_each_node=True,
        push_to_hub=True,
        hub_model_id=args.name_to_save,
        output_router_logits=True,
        router_aux_loss_coef=0.1,
    )

    trainer = SFTRegularizedSFTTrainer(
        model=model,
        args=args,
        train_dataset=dataset,  # type: ignore
    )
    print(trainer)
    print(dataset[0])
    print(trainer.train_dataset[0])
    trainer.train()
    print('done')


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", "-m", type=str, default="Qwen/Qwen3-1.7B-Base")
    parser.add_argument("--dataset", "-d", type=str, default="atrost/math_sft_40K_trl")
    parser.add_argument("--output-dir", "-o", type=str, default="outputs")
    parser.add_argument("--name-to-save", "-n", type=str, default="Qwen3-1.7B-Base-SFT-40K")
    parser.add_argument("--max-length", "-l", type=int, default=8192)
    parser.add_argument("--per-device-train-batch-size", "-b", type=int, default=2)
    parser.add_argument("--gradient-accumulation-steps", "-a", type=int, default=1)
    parser.add_argument("--learning-rate", "-r", type=float, default=2e-5)
    parser.add_argument("--num-train-epochs", "-e", type=int, default=3)
    parser.add_argument("--weight-decay", "-w", type=float, default=0.01)
    parser.add_argument("--max-grad-norm", "-g", type=float, default=0.1)
    parser.add_argument("--push-to-hub", "-p", type=bool, default=True)
    parser.add_argument("--use-vllm", type=bool, default=True)
    parser.add_argument("--vllm-host", type=str, default="0.0.0.0")
    parser.add_argument("--vllm-port", type=int, default=8000)
    args = parser.parse_args()
    main(args)
