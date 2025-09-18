from trl import SFTTrainer
from trl.trainer.utils import (
    entropy_from_logits,
    flush_left,
    generate_model_card,
    get_comet_experiment_url,
    pad,
    selective_log_softmax,
)
from transformers.models.auto.modeling_auto import (
    MODEL_FOR_CAUSAL_LM_MAPPING_NAMES,
    MODEL_MAPPING_NAMES,
)
from transformers.trainer import _is_peft_model
from trl.data_utils import (
    apply_chat_template,
    is_conversational_from_value,
    pack_dataset,
    prepare_multimodal_messages,
    truncate_dataset,
)
from datasets import Dataset, IterableDataset
from transformers import (
    AutoConfig,
    AutoProcessor,
    BaseImageProcessor,
    DataCollator,
    FeatureExtractionMixin,
    PreTrainedModel,
    PreTrainedTokenizerBase,
    ProcessorMixin,
    Trainer,
    TrainingArguments,
    is_wandb_available,
)
from trl.trainer.sft_config import SFTConfig
from trl.trainer.sft_trainer import remove_none_values, DataCollatorForLanguageModeling

import torch
import torch.nn as nn
from typing import Any, Optional, Union, Callable
from accelerate import PartialState, logging

logger = logging.get_logger(__name__)

class DataCollatorWithModelCompletion(DataCollatorForLanguageModeling):
    def torch_call(self, examples):
        out = super().torch_call(examples)
        examples_clone = examples.copy()
        for x in examples_clone:
            x["input_ids"] = x["model_completion_input_ids"]
            x["completion_mask"] = x["model_completion_mask"]
            if "model_completion_assistant_masks" in x:
                x["assistant_masks"] = x["model_completion_assistant_masks"]
        
        out_with_mc = super().torch_call(examples_clone)

        # combine out and out_with_mc
        out["model_completion_input_ids"] = out_with_mc["input_ids"]
        out["model_completion_attention_mask"] = out_with_mc["attention_mask"]
        out["model_completion_position_ids"] = out_with_mc["position_ids"]
        out["model_completion_labels"] = out_with_mc["labels"]

        return out

def maybe_convert_to_chatml(example: dict[str, list]) -> dict[str, list]: # adapted from trl
    """
    Convert a conversational dataset with fields `from` and `value` to ChatML format.

    This function modifies conversational data to align with OpenAI's ChatML format:
    - Replaces the key `"from"` with `"role"` in message dictionaries.
    - Replaces the key `"value"` with `"content"` in message dictionaries.
    - Renames `"conversations"` to `"messages"` for consistency with ChatML.

    Args:
        example (`dict[str, list]`):
            A single data entry containing a list of messages.

    Returns:
        `dict[str, list]`:
            Example reformatted to ChatML style.

    Example:
    ```python
    >>> from trl import maybe_convert_to_chatml

    >>> example = {
    ...     "conversations": [
    ...         {"from": "user", "value": "What color is the sky?"},
    ...         {"from": "assistant", "value": "It is blue."},
    ...     ]
    ... }
    >>> maybe_convert_to_chatml(example)
    {'messages': [{'role': 'user', 'content': 'What color is the sky?'},
                  {'role': 'assistant', 'content': 'It is blue.'}]}
    ```
    """
    # List of possible keys containing message lists
    for key in ["prompt", "completion", "chosen", "rejected", "messages", "conversations", "model_completion"]:
        if key in example and isinstance(example[key], list):
            messages = example[key]
            for message in messages:
                if isinstance(message, dict):
                    if "from" in message:
                        message["role"] = message.pop("from")
                    if "value" in message:
                        message["content"] = message.pop("value")

    # Rename "conversations" to "messages"
    if "conversations" in example:
        example["messages"] = example.pop("conversations")

    return example

def is_conversational(example: dict[str, Any]) -> bool: # adapted from trl
    r"""
    Check if the example is in a conversational format.

    Args:
        example (`dict[str, Any]`):
            A single data entry of a dataset. The example can have different keys depending on the dataset type.

    Returns:
        `bool`:
            `True` if the data is in a conversational format, `False` otherwise.

    Examples:

    ```python
    >>> example = {"prompt": [{"role": "user", "content": "What color is the sky?"}]}
    >>> is_conversational(example)
    True

    >>> example = {"prompt": "The sky is"}
    >>> is_conversational(example)
    False
    ```
    """
    supported_keys = ["prompt", "chosen", "rejected", "completion", "messages", "model_completion"]
    example_keys = {key for key in example.keys() if key in supported_keys}

    # It must have one of the supported keys
    if example_keys:
        key = example_keys.pop()  # take the first supported key
        maybe_messages = example[key]
        # It must be a list of messages
        if isinstance(maybe_messages, list):
            maybe_message = maybe_messages[0]
            # Each message must a list of dictionaries with keys "role" and "content"
            if isinstance(maybe_message, dict) and "role" in maybe_message and "content" in maybe_message:
                return True

    return False


class SFTRegularizedSFTTrainer(SFTTrainer):
    def __init__(self, aux_loss_coef, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.aux_loss_enabled = True
        self.aux_loss_coef = aux_loss_coef

        tokenizer = self.tokenizer
        use_flash_attention = self.model.config._attn_implementation in [
            "flash_attention_2",
            "flash_attention_3",
            "kernels-community/vllm-flash-attn3",
        ]

        pad_token = self.args.pad_token or tokenizer.pad_token or tokenizer.eos_token
        pad_token_id = tokenizer.convert_tokens_to_ids(pad_token)
        if pad_token_id is None:
            raise ValueError(
                f"The specified `pad_token` ('{pad_token}') is not found in the vocabulary of the given "
                f"`tokenizer` ({tokenizer.__class__.__name__}). Ensure that the `pad_token` exists "
                "in the vocabulary before using it as a padding token."
            )
        data_collator = DataCollatorWithModelCompletion(
                pad_token_id=pad_token_id,
                completion_only_loss=self.completion_only_loss,
                padding_free=self.padding_free,
                # Using position_ids without flash_attn hurts the training
                return_position_ids=use_flash_attention,
                pad_to_multiple_of=self.args.pad_to_multiple_of,
            )
        self.data_collator = data_collator

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        """
        Compute training loss and additionally compute token accuracies
        """
        mode = "train" if self.model.training else "eval"

        # Set aside labels as it will be dropped by super().compute_loss() if a custom `compute_loss_func` is used.
        # This can be removed when this issue is fixed.
        labels = inputs["labels"]

        # If not set, defaults from model config and may warn since cache isn't compatible with gradient checkpointing
        inputs["use_cache"] = False
        (loss, outputs) = super(SFTTrainer, self).compute_loss(
            model, inputs, return_outputs=True, num_items_in_batch=num_items_in_batch
        )
        inputs_clone = inputs.copy()
        inputs_clone["input_ids"] = inputs["model_completion_input_ids"]
        inputs_clone["attention_mask"] = inputs["model_completion_attention_mask"]
        inputs_clone["position_ids"] = inputs["model_completion_position_ids"]
        inputs_clone["labels"] = inputs["model_completion_labels"]
        (aux_loss, _) = super(SFTTrainer, self).compute_loss(
            model, inputs_clone, return_outputs=True, num_items_in_batch=num_items_in_batch
        )

        # Add auxiliary loss if available
        if self.aux_loss_enabled and self.aux_loss_coef:
            loss += self.aux_loss_coef * aux_loss
        else:
            raise ValueError("Auxiliary loss is not enabled")

        # Compute entropy
        if not self.args.use_liger_kernel:  # liger doesn't return logits
            with torch.no_grad():
                per_token_entropy = entropy_from_logits(outputs.logits)
                if "attention_mask" in inputs:
                    attention_mask = inputs["attention_mask"]
                    # When using Prompt Tuning, we need to add attention for the virtual tokens (all set to 1).
                    virtual_attention_mask = torch.ones(
                        attention_mask.size(0), self.num_virtual_tokens, device=attention_mask.device
                    )
                    attention_mask = torch.cat((virtual_attention_mask, attention_mask), dim=1)
                    entropy = torch.sum(per_token_entropy * attention_mask) / attention_mask.sum()
                elif "position_ids" in inputs:
                    entropy = torch.mean(per_token_entropy)
                else:
                    raise ValueError("Expected 'attention_mask' or 'position_ids' in inputs.")
                entropy = self.accelerator.gather_for_metrics(entropy).mean().item()
            self._metrics[mode]["entropy"].append(entropy)

        if mode == "train":
            # When using padding-free, the attention_mask is not present in the inputs, instead we have cu_seq_lens_q,
            # cu_seq_lens_k, and max_length_k, max_length_q and position_ids.
            if "attention_mask" in inputs:
                num_tokens_in_batch = self.accelerator.gather_for_metrics(inputs["attention_mask"].sum()).sum().item()
            elif "position_ids" in inputs:
                local_num_tokens = torch.tensor(inputs["position_ids"].size(1), device=inputs["position_ids"].device)
                num_tokens_in_batch = self.accelerator.gather_for_metrics(local_num_tokens).sum().item()
            else:
                raise ValueError("Expected 'attention_mask' or 'position_ids' in inputs.")
            self._total_train_tokens += num_tokens_in_batch
        self._metrics[mode]["num_tokens"] = [self._total_train_tokens]

        # Compute token accuracy if we have labels and if the model is not using Liger (no logits)
        if not self.args.use_liger_kernel:
            with torch.no_grad():
                if "shift_labels" in inputs:
                    # When using CP, labels are pre-shifted. We must use these (and cannot manually shift) because:
                    # - The first discarded token from inputs["labels"] actually belongs to process n-1
                    # - The last logits require the label from process n+1
                    shift_logits = outputs.logits.contiguous()
                    shift_labels = inputs["shift_labels"]
                else:
                    shift_logits = outputs.logits[..., :-1, :].contiguous()
                    shift_labels = labels[..., 1:].contiguous()

                # When using Prompt Tuning, skip the virtual tokens in logits before accuracy computation, since they do
                # not correspond to actual input labels.
                shift_logits = shift_logits[:, self.num_virtual_tokens :, :]

                # Get predictions
                predictions = shift_logits.argmax(dim=-1)

                # Create mask for non-padding tokens (assuming ignore_index is -100)
                mask = shift_labels != -100

                # Calculate accuracy only on non-padding tokens
                correct_predictions = (predictions == shift_labels) & mask
                total_tokens = mask.sum()
                correct_tokens = correct_predictions.sum()

                # Gather the correct_tokens and total_tokens across all processes
                correct_tokens = self.accelerator.gather_for_metrics(correct_tokens)
                total_tokens = self.accelerator.gather_for_metrics(total_tokens)

                # Compute the mean token accuracy and log it
                total_sum = total_tokens.sum()
                accuracy = (correct_tokens.sum() / total_sum).item() if total_sum > 0 else 0.0
                self._metrics[mode]["mean_token_accuracy"].append(accuracy)
                if self.aux_loss_enabled:
                    aux_loss = self.accelerator.gather_for_metrics(aux_loss).mean().item()
                    self._metrics[mode]["aux_loss"].append(aux_loss)

        return (loss, outputs) if return_outputs else loss

    def _set_signature_columns_if_needed(self):
        # If `self.args.remove_unused_columns` is True, non-signature columns are removed.
        # By default, this method sets `self._signature_columns` to the model's expected inputs (usually, "input_ids"
        # and "attention_mask"). When using `train_on_completion_only` we add a "completion_mask" column to the
        # dataset. So we need to override the default signature columns to include "completion_mask" as well.
        if self._signature_columns is None:
            if self._is_vlm:
                self._signature_columns = ["messages", "prompt", "completion", "images", "model_completion"]
            else:
                self._signature_columns = ["input_ids", "labels", "seq_lengths", "completion_mask", "assistant_masks", "model_completion_input_ids", "model_completion_mask", "model_completion_assistant_masks"]

    def _prepare_dataset(
        self,
        dataset: Union[Dataset, IterableDataset],
        processing_class: Union[PreTrainedTokenizerBase, BaseImageProcessor, FeatureExtractionMixin, ProcessorMixin],
        args: SFTConfig,
        packing: bool,
        formatting_func: Optional[Callable[[dict], str]],
        dataset_name: str,
    ) -> Union[Dataset, IterableDataset]:
        # Tabular backends like Arrow/Parquet insert `None` for mismatched keys in nested structures. Clean them from
        # sampled data.
        if isinstance(dataset, Dataset):  # IterableDataset does not support `with_transform`
            dataset = dataset.with_transform(remove_none_values)

        # If the dataset is already preprocessed (tokenized), skip the processing steps.
        column_names = list(next(iter(dataset)).keys())
        is_processed = "input_ids" in column_names

        # Build the kwargs for the `map` function
        map_kwargs = {}
        if isinstance(dataset, Dataset):  # IterableDataset does not support num_proc
            map_kwargs["num_proc"] = args.dataset_num_proc

        with PartialState().main_process_first():
            # Apply the formatting function if any
            if formatting_func is not None and is_processed:
                logger.warning(
                    "You passed a dataset that is already processed (contains an `input_ids` field) together with a "
                    "formatting function. Therefore `formatting_func` will be ignored. Either remove the "
                    "`formatting_func` or pass a dataset that is not already processed.",
                )

            if formatting_func is not None and not is_processed:
                if isinstance(dataset, Dataset):  # `IterableDataset.map` does not support `desc`
                    map_kwargs["desc"] = f"Applying formatting function to {dataset_name} dataset"

                def _func(example):
                    return {"text": formatting_func(example)}

                dataset = dataset.map(_func, batched=False, **map_kwargs)

            if not is_processed:
                # Convert the dataset to ChatML if needed
                first_example = next(iter(dataset))
                if is_conversational_from_value(first_example):
                    if isinstance(dataset, Dataset):  # `IterableDataset.map` does not support `desc`
                        map_kwargs["desc"] = f"Converting {dataset_name} dataset to ChatML"
                    column_names = next(iter(dataset)).keys()
                    dataset = dataset.map(
                        maybe_convert_to_chatml,
                        remove_columns="conversations" if "conversations" in column_names else None,
                        **map_kwargs,
                    )

                # Apply the chat template if needed
                first_example = next(iter(dataset))
                if not is_conversational(first_example):
                    if isinstance(dataset, Dataset):  # `IterableDataset.map` does not support `desc`
                        map_kwargs["desc"] = f"Adding EOS to {dataset_name} dataset"

                    def add_eos(example, eos_token):
                        if "text" in example and not example["text"].endswith(eos_token):  # language modeling case
                            example["text"] = example["text"] + eos_token
                        elif "completion" in example and not example["completion"].endswith(eos_token):
                            example["completion"] = example["completion"] + eos_token
                        if "model_completion" in example and not example["model_completion"].endswith(eos_token):
                            example["model_completion"] = example["model_completion"] + eos_token
                        return example

                    dataset = dataset.map(
                        add_eos,
                        fn_kwargs={"eos_token": processing_class.eos_token},
                        remove_columns="messages" if "messages" in column_names else None,  # renamed to "text"
                        **map_kwargs,
                    )

                # Tokenize the dataset
                if isinstance(dataset, Dataset):  # `IterableDataset.map` does not support `desc`
                    map_kwargs["desc"] = f"Tokenizing {dataset_name} dataset"

                def tokenize(example, processing_class, dataset_text_field, assistant_only_loss):
                    if "prompt" in example:  # prompt-completion case
                        output = {}
                        if is_conversational(example):
                            prompt_ids = processing_class.apply_chat_template(
                                example["prompt"],
                                tools=example.get("tools"),
                                **example.get("chat_template_kwargs", {}),
                            )
                            prompt_completion_processed = processing_class.apply_chat_template(
                                example["prompt"] + example["completion"],
                                return_dict=True,
                                return_assistant_tokens_mask=assistant_only_loss,
                                tools=example.get("tools"),
                                **example.get("chat_template_kwargs", {}),
                            )
                            prompt_model_completion_processed = processing_class.apply_chat_template(
                                example["prompt"] + example["model_completion"],
                                return_dict=True,
                                return_assistant_tokens_mask=assistant_only_loss,
                                tools=example.get("tools"),
                                **example.get("chat_template_kwargs", {}),
                            )
                            prompt_completion_ids = prompt_completion_processed["input_ids"]
                            prompt_model_completion_ids = prompt_model_completion_processed["input_ids"]
                            if "assistant_masks" in prompt_completion_processed:
                                output["assistant_masks"] = prompt_completion_processed["assistant_masks"]
                            if "assistant_masks" in prompt_model_completion_processed:
                                output["model_completion_assistant_masks"] = prompt_model_completion_processed["assistant_masks"]
                        else:
                            prompt_ids = processing_class(text=example["prompt"])["input_ids"]
                            prompt_completion_ids = processing_class(text=example["prompt"] + example["completion"])[
                                "input_ids"
                            ]
                            prompt_model_completion_ids = processing_class(text=example["prompt"] + example["model_completion"])[
                                "input_ids"
                            ]

                        # Check if the tokenized prompt starts with the tokenized prompt+completion
                        if not prompt_completion_ids[: len(prompt_ids)] == prompt_ids or not prompt_model_completion_ids[: len(prompt_ids)] == prompt_ids:
                            logger.warning(
                                "Mismatch between tokenized prompt and the start of tokenized prompt+completion. "
                                "This may be due to unexpected tokenizer behavior, whitespace issues, or special "
                                "token handling. Verify that the tokenizer is processing text consistently."
                            )

                        # Create a completion mask
                        completion_mask = [0] * len(prompt_ids) + [1] * (len(prompt_completion_ids) - len(prompt_ids))
                        model_completion_mask = [0] * len(prompt_ids) + [1] * (len(prompt_model_completion_ids) - len(prompt_ids))
                        output["input_ids"] = prompt_completion_ids
                        output["model_completion_input_ids"] = prompt_model_completion_ids
                        output["completion_mask"] = completion_mask
                        output["model_completion_mask"] = model_completion_mask

                    else:  # language modeling case
                        raise NotImplementedError("Language modeling case is not implemented")
                        if is_conversational(example):
                            processed = processing_class.apply_chat_template(
                                example["messages"],
                                return_dict=True,
                                return_assistant_tokens_mask=assistant_only_loss,
                                tools=example.get("tools"),
                                **example.get("chat_template_kwargs", {}),
                            )
                            if "assistant_masks" in processed and 1 not in processed["assistant_masks"]:
                                raise RuntimeError(
                                    "You're using `assistant_only_loss=True`, but at least one example has no "
                                    "assistant tokens. This usually means the tokenizer's chat template doesn't "
                                    "generate assistant masks — it may be missing the `{% generation %}` keyword. Please "
                                    "check the template and ensure it's correctly configured to support assistant "
                                    "masking."
                                )
                            output = {k: processed[k] for k in ("input_ids", "assistant_masks") if k in processed}
                        else:
                            output = {"input_ids": processing_class(text=example[dataset_text_field])["input_ids"]}
                    return output

                dataset = dataset.map(
                    tokenize,
                    fn_kwargs={
                        "processing_class": processing_class,
                        "dataset_text_field": args.dataset_text_field,
                        "assistant_only_loss": args.assistant_only_loss,
                    },
                    **map_kwargs,
                )

            # Pack or truncate
            if packing:
                if args.max_length is None:
                    raise ValueError("When packing is enabled, `max_length` can't be `None`.")
                if isinstance(dataset, Dataset):  # `IterableDataset.map` does not support `desc`
                    map_kwargs["desc"] = f"Packing {dataset_name} dataset"

                columns = ["input_ids", "model_completion_input_ids"]
                if "completion_mask" in dataset.column_names:
                    columns.append("completion_mask")
                if "model_completion_mask" in dataset.column_names:
                    columns.append("model_completion_mask")
                if "assistant_masks" in dataset.column_names:
                    columns.append("assistant_masks")
                if "model_completion_assistant_masks" in dataset.column_names:
                    columns.append("model_completion_assistant_masks")

                dataset = dataset.select_columns(columns)

                # Packing adds new column "seq_lengths" needed for document aware FlashAttention
                dataset = pack_dataset(dataset, args.max_length, args.packing_strategy, map_kwargs)
            elif args.max_length is not None:
                if isinstance(dataset, Dataset):  # `IterableDataset.map` does not support `desc`
                    map_kwargs["desc"] = f"Truncating {dataset_name} dataset"
                dataset = truncate_dataset(dataset, args.max_length, map_kwargs)
            # For Liger kernel, ensure only the essential columns
            if args.use_liger_kernel:
                collator_expected_keys = {"input_ids", "seq_lengths", "completion_mask", "assistant_masks", "model_completion_input_ids", "model_completion_mask", "model_completion_assistant_masks"}
                dataset = dataset.select_columns(collator_expected_keys.intersection(dataset.column_names))

        return dataset
