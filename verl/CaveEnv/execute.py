import os
import time
import json
from vllm import LLM, SamplingParams
from transformers import AutoTokenizer
import multiprocessing
from multiprocessing import Process, Queue
from multiprocessing import Pool
import pprint
import random
import queue
from verl.CaveEnv.env import CaveEnv
from verl.CaveEnv.lib import *
# from CaveEnv.prompt_guess import Prompt
# from CaveEnv.prompt_position import Prompt
from verl.CaveEnv.prompt_true_position import Prompt
# from CaveEnv.prompt_true_position_guess import Prompt
# from CaveEnv.prompt import Prompt

def load_model(params=None):
    """load model and tokenizer"""
    if params is None:
        params = {
            "model_path": "/home/ubuntu/my_models_train/sft/cot_pos_distill/epoch_12",
            "tensor_parallel_size": 1,
            "gpu_memory_utilization": 0.9,
            "swap_space": 16,
            "block_size": 32,
            "trust_remote_code": False,
            "dtype": "bfloat16",
            "enforce_eager": False,
            "max_num_seqs": 128,
            # task="generate",
            # max_model_len=MAX_TOKENS,
            # quantization="awq",
        }

    model =  LLM(
        model=params["model_path"],
        tensor_parallel_size=params["tensor_parallel_size"],
        gpu_memory_utilization=params["gpu_memory_utilization"],
        swap_space=params["swap_space"],
        block_size=params["block_size"],
        trust_remote_code=params["trust_remote_code"],
        dtype=params["dtype"],
        enforce_eager=params["enforce_eager"],
        max_num_seqs=params["max_num_seqs"],
    )

    return model

def generate_response(model, formatted_messages, params=None):
    """  generate response  """
    if params is None:
        params = {
            "temperature": 0.7,
            "top_k": 50,
            "top_p": 0.7,
            "max_response_length": 4000,
            "ignore_eos": False,
            "stop": ["<|im_end|>", "<|endoftext|>"],
            "seed": 0,
        }
    sampling_params = SamplingParams(
        temperature=params["temperature"],
        top_p=params['top_p'],
        top_k=params['top_k'],
        max_tokens=params['max_response_length'],
        ignore_eos=params['ignore_eos'],
        stop=params['stop'] ,  # 停止标识符，不会输出
        seed=params['seed'],
    )
    responses = model.generate(formatted_messages, sampling_params)
    return responses

def init_env(env_config):
    # init env
    previous_pos = "This is the first step so there is no Previous positions."
    env = CaveEnv(env_config=env_config, render_mode="rgb_array")
    env.reset()

    # get init prompt
    pro, temp = env.get_obs_output_llm()
    temp["Previous positions"] = previous_pos
    obs_prompt = pro + pprint.pformat(temp, indent=2, sort_dicts=False, width=200)
    return env, obs_prompt

def step_env(env, action):
    if not env.done:
        #  step env
        previous_pos = pprint.pformat(env.history_pos, indent=2, sort_dicts=False, width=200)
        obs, reward, terminated, truncated, info = env.step(action)

        #  get prompt
        pro, temp = env.get_obs_output_llm()
        temp["Previous positions"] = previous_pos
        obs_prompt = pro + pprint.pformat(temp, indent=2, sort_dicts=False, width=200)
        return obs_prompt, reward
    else:
        return "Env is over", 0

def process_obs_prompt(obs_prompts, tokenizer):
    env_prompt = Prompt()
    input_messages = []
    for obs_prompt in obs_prompts:
        input_message = env_prompt.wrap_message(obs_prompt)
        input_messages.append(input_message)

    # Apply chat template
    formatted_messages = tokenizer.apply_chat_template(
        input_messages,
        tokenize=False,
        add_generation_prompt=True
    )
    return formatted_messages

def get_action(response_all):
    action_str_maps = ['<MoveUp>', '<MoveDown>', '<MoveLeft>', '<MoveRight>']
    actions = []
    random.seed()
    for response in response_all:
        text = response.outputs[0].text
        action_match = extract_action(text)
        action_str = action_match if action_match else random.choice(action_str_maps)
        actions.append(action_str)
    return actions



def main():
    model = load_model()
    tokenizer = AutoTokenizer.from_pretrained("/home/ubuntu/my_models/QwQ-32B")

    num_envs = 1  # 假设我们需要5个环境
    env_configs = []
    for num_iter in range(num_envs):
        env_configs.append({"size": 3,
                            "num_pits": 1,
                            "max_steps": 20,
                            "start_pos": [0, 0],
                            "seed": num_iter,
                            "env_id": num_iter
                            })

    active_envs = set()
    obs_prompt_all = []
    for k in range(num_envs):
        env, obs_prompt = init_env(env_configs[k])
        active_envs.add(env)
        obs_prompt_all.append(obs_prompt)

    while active_envs:
        #  step 1  get message
        message_all = process_obs_prompt(obs_prompt_all, tokenizer)
        obs_prompt_all = []

        #  step 2  get response and action
        response_all = generate_response(model, message_all)
        action_all = get_action(response_all)
        # action_all = get_action(message_all)

        #  step 3  env step
        to_remove = []
        for k, env in enumerate(active_envs):
            action = action_all[k]
            obs_prompt, reward = step_env(env, action)
            # print(f'env: {env.env_id} reward:{reward}')
            if not env.done:
                obs_prompt_all.append(obs_prompt)
            else:
                to_remove.append(env)
                print(f'die_env：{env.env_id}')

        #  step 4  remove done env
        for die_env in to_remove:
            active_envs.remove(die_env)




if __name__ == "__main__":
    main()



