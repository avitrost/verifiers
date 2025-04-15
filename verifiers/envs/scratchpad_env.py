from abc import abstractmethod
from copy import deepcopy
from concurrent.futures import ThreadPoolExecutor
import random
import time
from typing import List, Dict, Sequence, Any, Union, Tuple

from datasets import Dataset
from pydantic import BaseModel
from ..imports import LLM, SamplingParams  # type: ignore
from verifiers.inference.vllm_client import VLLMClient
from verifiers.prompts import SCRATCHPAD_PROMPT, FIRST_SCRATCHPAD_PROMPT
from verifiers.rubrics import MathVerifyRubric
from verifiers import RewardFunc

from verifiers.envs.environment import Environment
from verifiers.utils import format_dataset, extract_solution

class ChatOutput(BaseModel):
    token_ids: List[int]
    text: str

class ChatResponseItem(BaseModel):
    prompt_token_ids: List[int]
    outputs: List[ChatOutput]

class ChatResponse(BaseModel):
    responses: List[ChatResponseItem]

def dict_to_chat_response(data: Dict[str, Any]) -> ChatResponse:
    """
    Recursively convert a dictionary to a ChatResponse object
    """
    # First, convert all outputs to ChatOutput objects
    if "responses" in data:
        for i, response_item in enumerate(data["responses"]):
            if "outputs" in response_item:
                data["responses"][i]["outputs"] = [
                    ChatOutput(**output) for output in response_item["outputs"]
                ]
        
        # Then convert all response items to ChatResponseItem objects
        data["responses"] = [ChatResponseItem(**item) for item in data["responses"]]
    
    # Finally, convert the entire dict to a ChatResponse object
    return ChatResponse(**data)

class ScratchpadEnv(Environment):
    def __init__(self,
                 dataset: Dataset | None = None,
                 eval_dataset: Dataset | None = None,
                 system_prompt: str = SCRATCHPAD_PROMPT,
                #  few_shot: List[Dict[str, str]] = [],
                 sampling_args: Dict[str, Any] = {},
                 mask_env_response: bool = True,
                 max_workers: int = 1,
                 max_steps: int = 10,
                 sleep_time: float = 1.0,
                 max_tries: int = 5,
                 **kwargs):
        super().__init__(**kwargs)
        self.system_prompt = system_prompt
        self.max_tries = max_tries
        self.rubric = MathVerifyRubric()
        self.verifier_func = self.rubric.get_reward_funcs()[0]  # TODO: edit if i have more
        # self.few_shot = few_shot
        if dataset is not None:
            self.dataset = format_dataset(
                dataset=dataset,
                system_prompt=self.system_prompt,
                # few_shot=self.few_shot
            )
        else:
            self.dataset = None
        if eval_dataset is not None:
            self.eval_dataset = format_dataset(
                dataset=eval_dataset,
                system_prompt=self.system_prompt,
                # few_shot=few_shot
            )
        else:   
            self.eval_dataset = None
        self.sampling_args = {
            "skip_special_tokens": False,
            "spaces_between_special_tokens": False,
            "n": 1,
            "stop": ["</attempt>"],
            "include_stop_str_in_output": True
        }
        self.sampling_args.update(sampling_args)
        self.env_mask = 0 if mask_env_response else 1
        self.max_workers = max_workers
        self.sleep_time = sleep_time
        self.max_steps = max_steps

    def get_reward_funcs(self, **kwargs: Any) -> List[RewardFunc]:
        return self.rubric.get_reward_funcs()
    
    def get_reward_weights(self, **kwargs: Any) -> List[float]:
        return self.rubric.get_reward_weights()

    def get_dataset(self, n: int = -1, seed: int = 0, **kwargs: Any) -> Dataset | None:
        if n > 0 and self.dataset is not None:
            return self.dataset.shuffle(seed=seed).select(range(n)) # type: ignore
        return self.dataset

    def get_eval_dataset(self, n: int = -1, seed: int = 0, **kwargs: Any) -> Dataset | None:
        if n > 0 and self.eval_dataset is not None:
            return self.eval_dataset.shuffle(seed=seed).select(range(n)) # type: ignore
        return self.eval_dataset
    
    def count_responses(self, messages: List[Dict[str, str]]) -> int:
        """
        Count the number of responses in the messages.
        """
        count = 0
        # print('-------------------------')
        # print(messages)
        # print('-------------------------')
        for message in messages:
            if message[-1]["role"] == "assistant":
                count += 1
        return count
    
    def is_completed(self, messages: List[Dict[str, str]], answer: str, **kwargs: Any) -> bool:
        response = messages[-1][-1]["content"]
        is_correct = self.verifier_func(response, answer)
        is_final = self.count_responses(messages) >= self.max_tries
        completed = is_correct or is_final
        return completed

    def extract_context(self, messages: List[Dict[str, str]]) -> str:
        """
        Extract the context from the messages.
        """
        message_string = messages[-1][-1]["content"]
        try:
            # Find the start and end of the attempt block
            start_idx = message_string.find("<attempt>")
            end_idx = message_string.find("</attempt>")
            
            if start_idx != -1 and end_idx != -1:
                # Extract the content between the tags, excluding the tags themselves
                context = message_string[start_idx + len("<attempt>"):end_idx].strip()
            else:
                # If tags not found, return empty string
                raise ValueError("Attempt tags not found in message string.")
            
            return context
        except Exception as e:
            print(f"Error extracting context: {e}")
            return ""
    
    def create_new_prompt(self, messages: List[Dict[str, str]], original_prompt: str) -> str:  # TODO: add try number? 
        context = self.extract_context(messages)
        new_prompt = "<previous_attempts>\n" + context + "\n<\previous_attempts>\n" + original_prompt
        return new_prompt

    def env_response(self, messages: List[Dict[str, str]], original_prompt: str, **kwargs: Any) -> Dict[str, str]:
        content = self.create_new_prompt(messages, original_prompt)
        new_prompt = [{"role": "user", "content": content}]
        return new_prompt

    def step(self,
             states: List[Dict[str, Any]],
             llm: LLM | VLLMClient,
             sampling_params: SamplingParams) -> List[Dict[str, Any]]:
        
        live_indices = [i for i, s in enumerate(states) if not s["completed"]]

        # get the most recent prompt only
        messages_to_step = [states[i]["messages"][-1] for i in live_indices]


        if isinstance(llm, VLLMClient):
            llm_responses = llm.chat(
                messages_to_step,
                n=1,
                repetition_penalty=sampling_params.repetition_penalty,
                temperature=sampling_params.temperature,
                top_p=sampling_params.top_p,
                top_k=sampling_params.top_k,
                min_p=sampling_params.min_p,
                max_tokens=sampling_params.max_tokens, # type: ignore
                stop=sampling_params.stop, # type: ignore
                include_stop_str_in_output=sampling_params.include_stop_str_in_output,
                skip_special_tokens=sampling_params.skip_special_tokens,
                spaces_between_special_tokens=sampling_params.spaces_between_special_tokens
            ) # type: ignore
            llm_responses = dict_to_chat_response(llm_responses).responses
        else:
            llm_responses = llm.chat(messages_to_step, sampling_params=sampling_params, use_tqdm=False) # type: ignore

        #for i, j in enumerate(live_indices):
        def update_state(j, llm_response):
            # sleep for 0-1 seconds to avoid rate limiting
            # time.sleep(self.sleep_time * random.random())

            state = deepcopy(states[j])
            state["prompt_ids"].append(llm_response.prompt_token_ids)
            state["messages"][-1].append({"role": "assistant", "content": llm_response.outputs[0].text})
        
            # get token lengths of env response and new completion
            # total_prev_len = len(state["prompt_ids"][-1]) # + len(state["completion_ids"])
            # env_response_len  = 0 # len(list(llm_response.prompt_token_ids)) - total_prev_len # type: ignore
            # new_completion_len = len(llm_response.outputs[0].token_ids)

            # update completion ids
            new_completion_ids = list(llm_response.prompt_token_ids) # type: ignore
            new_completion_ids.extend(list(llm_response.outputs[0].token_ids))
            new_completion_ids = new_completion_ids[len(state["prompt_ids"][-1]):]

            if new_completion_ids[-1] != 198 and new_completion_ids[-2] != self.message_end_id:
                new_completion_ids.append(self.message_end_id)
                new_completion_ids.append(198)
                # state["completion_mask"][-1].append(1)
                # state["completion_mask"][-1].append(1)

            # if len(state["completion_ids"]) > len(state["completion_mask"][-1]): # type: ignore
            #     state["completion_mask"][-1].extend([1] * (len(state["completion_ids"]) - len(state["completion_mask"][-1]))) # type: ignore
            # if len(state["completion_mask"][-1]) > len(state["completion_ids"]): # type: ignore
            #     state["completion_mask"][-1] = state["completion_mask"][-1][:len(state["completion_ids"])] # type: ignore
            
            if self.is_completed(state["messages"], state["answer"]): # or len(state["completion_ids"]) > sampling_params.max_tokens - 1: # type: ignore
                state["completed"] = True
                # state["completion_ids"] = state["completion_ids"][:sampling_params.max_tokens] TODO: check this
                # state["completion_mask"][-1] = state["completion_mask"][-1][:len(state["completion_ids"])]
            else:
                state["messages"].append(self.env_response(state["messages"], state["original_prompt"]))

            # enforce that the completion mask and completion ids are the same length
            # weird bug that happens rarely and only for certain models; something tokenizer related :(
            # if not len(state["completion_mask"][-1]) == len(state["completion_ids"]):
            #     print(state["messages"])
            #     print(state["completion_mask"])
            #     print(state["completion_ids"])
            #     min_len = min(len(state["completion_mask"]), len(state["completion_ids"]))
            #     state["completion_mask"] = state["completion_mask"][:min_len]
            #     state["completion_ids"] = state["completion_ids"][:min_len]

            state["completion_ids"].append(new_completion_ids)
            state["completion_mask"].append([1] * len(state["completion_ids"][-1]))

            return j, state

        # Original threaded version
        # with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
        #     results = list(executor.map(
        #         lambda args: update_state(*args),
        #         [(j, llm_responses[i]) for i, j in enumerate(live_indices)]
        #     ))
        
        # Non-threaded version
        results = []
        for i, j in enumerate(live_indices):
            result = update_state(j, llm_responses[i])
            results.append(result)

        for j, state in results:
            states[j] = state

        return states

    def generate(self, prompts: List[List[Dict[str, Any]]],
                 answers: List[str],
                 llm: LLM | VLLMClient,  
                 sampling_params: SamplingParams,
                 **kwargs: Any) -> Dict[str, List[Sequence[int]] | List[str] |  List[List[Dict[str, Any]]]]:
        custom_sp = sampling_params.clone()
        for k, v in self.sampling_args.items():
            setattr(custom_sp, k, v)

        # initialize state variables
        all_completed = False
        states = [{
            "messages": [m],
            "prompt_messages": len(m), # TODO: ?
            "prompt_ids": [],
            "completed": False,
            "completion_ids": [],
            "completion_mask": [],
            "answer": answer,
            "original_prompt": deepcopy(m[0]["content"]),  # Store copy of prompt content
        } for m, answer in zip(prompts, answers)]

        # main loop
        while not all_completed:
            states = self.step(states, llm, custom_sp)
            all_completed = all(state["completed"] for state in states)

        completion_messages = [s["messages"][s["prompt_messages"]:] for s in states]
        completion_ids = [s["completion_ids"] for s in states]
        completion_mask = [s["completion_mask"] for s in states]
        output = {
            "ids": completion_ids,  # list of lists of ids
            "messages": completion_messages, # list of lists of [p, c] ??????
            "mask": completion_mask # list of lists of 
        }
        return output

    
    