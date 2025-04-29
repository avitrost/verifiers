import warnings
from typing import Callable, Optional, Union, Any, List

from accelerate.utils import broadcast_object_list, gather, gather_object
from datasets import Dataset, IterableDataset
from peft import PeftConfig # type: ignore
import torch
import numpy as np
from torch import nn
from transformers import (
    PreTrainedModel,
    PreTrainedTokenizerBase,
    Trainer,
    TrainerCallback,
    is_wandb_available
)
from verifiers import RewardFunc
from verifiers.envs.environment import Environment
from verifiers.utils.logging_utils import print_prompt_completions_sample
from verifiers.imports import LLM, SamplingParams
from verifiers.inference.vllm_client import VLLMClient

# monkey patch vllm client
import trl.extras.vllm_client
trl.extras.vllm_client.VLLMClient = VLLMClient

from trl import GRPOTrainer, GRPOConfig
from trl.data_utils import maybe_apply_chat_template
from trl.import_utils import is_rich_available
from trl.trainer.utils import pad

if is_wandb_available():
    import wandb



# torch.nanstd doesn't exist, so we define it here
def nanstd(tensor: torch.Tensor) -> torch.Tensor:
    """
    Compute the standard deviation of a tensor, ignoring NaNs. This function only supports 1D tensors.

    Args:
        tensor (`torch.Tensor`):
            Input tensor of shape `(N,)`.

    Returns:
        `torch.Tensor`:
            Standard deviation of the tensor, ignoring NaNs.
    """
    variance = torch.nanmean((tensor - torch.nanmean(tensor, keepdim=True)) ** 2)  # Compute variance ignoring NaNs
    count = torch.sum(~torch.isnan(tensor))  # Count of non-NaN values
    variance *= count / (count - 1)  # Bessel's correction
    return torch.sqrt(variance)

class GRPOScratchpadEnvTrainer(GRPOTrainer):
    def __init__(
            self,
            model: Union[str, PreTrainedModel],
            env: Environment,
            reward_funcs: Union[RewardFunc, list[RewardFunc]],
            scale_rewards: bool = False,
            args: Optional[GRPOConfig] = None,
            train_dataset: Optional[Union[Dataset, IterableDataset]] = None,
            eval_dataset: Optional[Union[Dataset, IterableDataset]] = None,
            processing_class: Optional[PreTrainedTokenizerBase] = None,
            callbacks: Optional[list[TrainerCallback]] = None,
            optimizers: tuple[Optional[torch.optim.Optimizer], Optional[torch.optim.lr_scheduler.LambdaLR]] = (None, None),
            peft_config: Optional["PeftConfig"] = None,
            **kwargs,
    ):
        self.vllm_client = None
        if not args.use_vllm: # type: ignore
            raise ValueError("vLLM must be enabled for GRPOEnvTrainer")
        if not (callable(reward_funcs) or (isinstance(reward_funcs, list) and all(callable(f) for f in reward_funcs))): 
            raise ValueError("reward_funcs must be a function or a list of functions. Use vLLM to host neural reward models.")
        
        super().__init__(
            model=model,
            reward_funcs=reward_funcs,
            args=args,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            processing_class=processing_class,
            callbacks=callbacks,
            optimizers=optimizers,
            peft_config=peft_config,
            **kwargs,
        )
        self.env = env
        self.scale_rewards = scale_rewards
        self.sampling_params = SamplingParams(
            max_tokens=self.max_completion_length,
            temperature=self.temperature,
            top_p=self.top_p,
            top_k=-1 if self.top_k is None else self.top_k,
            min_p=0.0 if self.min_p is None else self.min_p,
            repetition_penalty=self.repetition_penalty
        )

    def _generate_and_score_completions(
         self, inputs: dict[str, Union[torch.Tensor, Any]]   
    ) -> dict[str, Union[torch.Tensor, Any]]:
        device = self.accelerator.device
        prompts = [x["prompt"] for x in inputs] # type: ignore
        answers = [x["answer"] for x in inputs] # type: ignore
        prompts_text = [maybe_apply_chat_template(example, self.processing_class)["prompt"] for example in inputs] # type: ignore
        prompt_inputs = self.processing_class(
            prompts_text, return_tensors="pt", padding=True, padding_side="left", add_special_tokens=False # type: ignore
        ) # type: ignore
        prompt_inputs = Trainer._prepare_inputs(self, prompt_inputs) # type: ignore
        prompt_ids, prompt_mask = prompt_inputs["input_ids"], prompt_inputs["attention_mask"]

        if self.max_prompt_length is not None:
            prompt_ids = prompt_ids[:, -self.max_prompt_length :]
            prompt_mask = prompt_mask[:, -self.max_prompt_length :]

        if self.state.global_step != self._last_loaded_step:
            self._move_model_to_vllm()
            self._last_loaded_step = self.state.global_step

        # Gather the original prompts in message dict form, not the text form
        all_prompts = gather_object(prompts)
        all_answers = gather_object(answers)
        if self.accelerator.is_main_process:
            env_result = self.env.generate(
                prompts=all_prompts,
                answers=all_answers,
                llm=self.vllm_client, # type: ignore
                sampling_params=self.sampling_params,
            )

            # will be lists
            completion_ids = env_result['ids']
            completion_messages = env_result['messages']
            completion_mask = env_result['mask']
            num_tries = env_result['num_tries']

        else:
            completion_ids = [None] * len(all_prompts)
            completion_messages = [None] * len(all_prompts)
            completion_mask = [None] * len(all_prompts)

        completion_ids = broadcast_object_list(completion_ids, from_process=0)
        completion_messages = broadcast_object_list(completion_messages, from_process=0)
        completion_mask = broadcast_object_list(completion_mask, from_process=0)


        process_slice = slice(
            self.accelerator.process_index * len(prompts),
            (self.accelerator.process_index + 1) * len(prompts),
        )

        total_completion_ids = completion_ids[process_slice]
        total_completion_messages = completion_messages[process_slice]
        total_completion_mask = completion_mask[process_slice]

        # Pad + mask after per-sequence EOS tokens

        lst_old_per_token_logps = []
        lst_ref_per_token_logps = []
        lst_completion_ids = []
        lst_completion_mask = []

        print('-----------------------')
        print(f"total_completion_ids: {total_completion_ids}")
        print(f"total_completion_messages: {total_completion_messages}")
        print(f"total_completion_mask: {total_completion_mask}")
        print('-----------------------')
        # Init completions
        completions = []
        for x in total_completion_messages:
            completions.append(x[0])
        
        for i in range(self.env.max_tries):
            tries_mask = np.ones(len(prompt_ids))
            # input('i = ' + str(i))

            # completion_ids = [x[i] for x in total_completion_ids]
            completion_ids = []
            if len(total_completion_ids) == 0:
                print('***********')
                print('prompts: ', prompts)
                print('i: ', i)
                print('env_result: ', env_result)
                print('))))))))))))))))))')
                input()
            for idx, x in enumerate(total_completion_ids):
                if i >= num_tries[idx]:
                    print('-----------------------')
                    print(f"total_completion_ids: {total_completion_ids}")
                    print(f"total_completion_messages: {total_completion_messages}")
                    print(f"total_completion_mask: {total_completion_mask}")
                    print(f"x: {x}")
                    print(f"i: {i}")
                    print(f"num_tries: {num_tries}")
                    print(f"idx: {idx}")
                    print('-----------------------')
                    tries_mask[idx] = 0
                    continue
                else:
                    completion_ids.append(x[i])
            completion_messages = []
            for idx, x in enumerate(total_completion_messages):
                print('-----------------------')
                # input("Press Enter to continue...")
                print(f"total_completion_messages: {total_completion_messages}")
                print(f"x: {x}")
                print(f"i: {i}")
                print(f"num_tries: {num_tries}")
                print(f"idx: {idx}")
                if i >= num_tries[idx]:
                    continue
                else:
                    completion_messages.append(x[i])
                    completions[idx] = x[i]
            completion_mask = []
            for idx, x in enumerate(total_completion_mask):
                if i >= num_tries[idx]:
                    continue
                else:
                    completion_mask.append(x[i])
            # completion_messages = [x[i] for x in total_completion_messages]
            # completion_mask = [x[i] for x in total_completion_mask]
            
            print('##############################')
            print(f"completion_ids: {completion_ids}")
            print("prompt_ids: ", prompt_ids)
            
            if len(completion_ids) == 0:
                print('*********** completion 0 ***********')
                print('prompts: ', prompts)
                print('i: ', i)
                print('env_result: ', env_result)
                print('))))))))))))))))))')
                input()

            print('MASK')
            print(tries_mask)

            completion_ids = [torch.tensor(ids, device=device) for ids in completion_ids]
            completion_ids = pad(completion_ids, padding_value=self.processing_class.pad_token_id) # type: ignore

            completion_mask = [torch.tensor(mask, device=device) for mask in completion_mask]
            completion_mask = pad(completion_mask, padding_value=0)

            print('&&&&&&&&&&&&&&&&&&&&')
            print(f"completion_ids: {completion_ids}")
            print("prompt_ids: ", prompt_ids)
            print("inputs: ", inputs)
            print(f"completion_ids size: {completion_ids.size()}")
            print("prompt_ids size: ", prompt_ids.size())
            prompt_completion_ids = torch.cat([prompt_ids[tries_mask == 1], completion_ids], dim=1)
            attention_mask = torch.cat([prompt_mask[tries_mask == 1], completion_mask], dim=1) # (B, P+C)
        
            # TODO: check this
            print('-----------------------')
            print("completion_ids: ", completion_ids)
            print("completion_ids size: ", completion_ids.size())
            print("completion_ids size 1: ", completion_ids.size(1))
            print(len(completion_ids))
            print(len(completion_ids[0]))
            print(completion_ids[0].size())
            if completion_ids.size(1) == 0:
                print('***********')
                print('prompts: ', prompts)
                print('i: ', i)
            print('&&&&&&&&&&&&&&&&&&&&&')
            logits_to_keep = completion_ids.size(1)

            # TODO: split up and concatenate the per token logps, result will be same dims as original(?). everything else same after?
            with torch.no_grad():
                # When using num_iterations == 1, old_per_token_logps == per_token_logps, so we can skip it's
                # computation here, and use per_token_logps.detach() instead.
                if self.num_iterations > 1:
                    print('-----------------------')
                    print(f"prompt_completion_ids: {prompt_completion_ids}")
                    print(f"attention_mask: {attention_mask}")
                    print(f"logits_to_keep: {logits_to_keep}")
                    print(f"self.model: {self.model}")
                    print("---------------------------")
                    old_per_token_logps = self._get_per_token_logps(
                        self.model, prompt_completion_ids, attention_mask, logits_to_keep
                    )
                else:
                    old_per_token_logps = None

                if self.beta == 0.0:
                    ref_per_token_logps = None
                elif self.ref_model is not None:
                    ref_per_token_logps = self._get_per_token_logps(
                        self.ref_model, prompt_completion_ids, attention_mask, logits_to_keep
                    )
                else:
                    with self.accelerator.unwrap_model(self.model).disable_adapter():
                        ref_per_token_logps = self._get_per_token_logps(
                            self.model, prompt_completion_ids, attention_mask, logits_to_keep
                        )
            lst_old_per_token_logps.append(old_per_token_logps)
            lst_ref_per_token_logps.append(ref_per_token_logps)
            lst_completion_ids.append(completion_ids)
            lst_completion_mask.append(completion_mask)
        
        # use message dicts for reward function inputs
        # completions = completion_messages  # this is the final iterate so this is correct to put outside loop (assuming the num tries is constant for all)
        print(completions)
        input()

        rewards_per_func = torch.zeros(len(prompts), len(self.reward_funcs), device=device)
        for i, reward_func in enumerate(self.reward_funcs):
            # Repeat all input columns (but "prompt" and "completion") to match the number of generations
            keys = [key for key in inputs[0] if key not in ["prompt", "completion"]] # type: ignore
            reward_kwargs = {key: [example[key] for example in inputs] for key in keys} # type: ignore
            output_reward_func = reward_func(prompts=prompts, completions=completions, **reward_kwargs) # type: ignore
            
            output_reward_func = [reward if reward is not None else torch.nan for reward in output_reward_func]
            rewards_per_func[:, i] = torch.tensor(output_reward_func, dtype=torch.float32, device=device)

        # If all reward functions return None for a given row, issue a detailed warning
        if torch.isnan(rewards_per_func).all(dim=1).any():
            nan_row_idx = torch.isnan(rewards_per_func).all(dim=1).nonzero(as_tuple=True)[0][0]
            row_reward_kwargs = {key: value[nan_row_idx] for key, value in reward_kwargs.items()} # type: ignore
            row_reward_kwargs["prompt"] = prompts[nan_row_idx]
            row_reward_kwargs["completion"] = completions[nan_row_idx] # type: ignore
            warnings.warn(
                f"All reward functions returned None for the following kwargs: {row_reward_kwargs}. "
                "Please ensure that at least one reward function returns a valid reward."
            )


        rewards_per_func = gather(rewards_per_func)

        # Apply weights to each reward function's output and sum
        rewards = (rewards_per_func * self.reward_weights.to(device).unsqueeze(0)).nansum(dim=1)

        # Compute grouped-wise rewards
        mean_grouped_rewards = rewards.view(-1, self.num_generations).mean(dim=1) # type: ignore

        # Normalize the rewards to compute the advantages
        mean_grouped_rewards = mean_grouped_rewards.repeat_interleave(self.num_generations, dim=0) # type: ignore
        advantages = (rewards - mean_grouped_rewards)
        
        std_grouped_rewards = rewards.view(-1, self.num_generations).std(dim=1) # type: ignore
        std_grouped_rewards = std_grouped_rewards.repeat_interleave(self.num_generations, dim=0) # type: ignore
        if self.scale_rewards:
            # Scale the rewards to be between 0 and 1
            advantages = advantages / (std_grouped_rewards + 1e-4)

        # Slice to keep only the local part of the data
        process_slice = slice(
            self.accelerator.process_index * len(prompts),
            (self.accelerator.process_index + 1) * len(prompts),
        )
        advantages = advantages[process_slice]

        # Log the metrics
        mode = "eval" if self.control.should_evaluate else "train"

        completion_length = self.accelerator.gather_for_metrics(completion_mask.sum(1)).float().mean().item() # type: ignore
        self._metrics[mode]["completion_length"].append(completion_length)

        # Calculate mean reward per function, but only for samples where the function was applied
        for i, reward_func in enumerate(self.reward_funcs):
            reward_func_name = reward_func.__name__ # type: ignore  
            # Only calculate mean for samples where this reward function was applied (non-NaN values)
            mean_rewards = torch.nanmean(rewards_per_func[:, i]).item()
            self._metrics[mode][f"rewards/{reward_func_name}"].append(mean_rewards)
            std_rewards = nanstd(rewards_per_func[:, i]).item()
            self._metrics[mode][f"rewards/{reward_func_name}/std"].append(std_rewards)
        self._metrics[mode]["reward"].append(rewards.mean().item())
        self._metrics[mode]["reward_std"].append(std_grouped_rewards.mean().item()) # type: ignore

        if self.log_completions and self.state.global_step % self.args.logging_steps == 0:
            prompts_to_log = gather_object(prompts)
            completions_to_log = gather_object(completions)
            rewards_to_log = rewards.tolist()

            if self.accelerator.is_main_process:
                if is_rich_available():
                    print_prompt_completions_sample(
                        [str(prompts_to_log[0][-1]["content"])],
                        [completions_to_log[0]],
                        [rewards_to_log[0]],
                        self.state.global_step,
                    )
                if self.args.report_to and "wandb" in self.args.report_to and wandb.run is not None: # type: ignore
                    import pandas as pd

                    # For logging
                    table = {
                        "step": [str(self.state.global_step)] * len(rewards),
                        "prompt": prompts_to_log,
                        "completion": completions_to_log,
                        "reward": rewards.tolist(),
                    }
                    df = pd.DataFrame(table)
                    wandb.log({"completions": wandb.Table(dataframe=df)}) # type: ignore

        concatenated_old_per_token_logps = torch.cat(lst_old_per_token_logps, dim=-1) if self.num_iterations > 1 else None
        concatenated_ref_per_token_logps = torch.cat(lst_ref_per_token_logps, dim=-1)
        concatenated_completion_ids = torch.cat(lst_completion_ids, dim=-1)
        concatenated_completion_mask = torch.cat(lst_completion_mask, dim=-1)
        
        return {
            "prompt_ids": prompt_ids,
            "prompt_mask": prompt_mask,
            "completion_ids": concatenated_completion_ids,
            "completion_mask": concatenated_completion_mask,
            "old_per_token_logps": concatenated_old_per_token_logps,
            "ref_per_token_logps": concatenated_ref_per_token_logps,
            "advantages": advantages,
        }