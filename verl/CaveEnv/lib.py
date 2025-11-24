


import os
import re
import glob
import json
import time
from queue import Empty
from matplotlib import pyplot as plt
from verl.CaveEnv.env import CaveEnv


def extract_action(text):
  # pattern = r"(?i)guess[\s\S]*?Action[\s\S]*?(MoveUP|MoveDown|MoveLeft|MoveRight)"
  text = text[-200:]
  pattern = r"(?i)[\s\S]*?Action[\s\S]*?(MoveUP|MoveDown|MoveLeft|MoveRight)"
  matches = re.findall(pattern, text, re.DOTALL)
  # match = match.lower()
  if matches:
     last_match = matches[-1].strip()
     return f"<{last_match}>"
  else:
     return None


def extract_previous_positions(text):
    # 第一次提取
    pattern1 = re.compile(r"visited breeze positions\s*.*?\s*\[(.*?)\]", re.IGNORECASE)
    pattern2 = re.compile(r"visited glitter positions\s*.*?\s*\[(.*?)\]", re.IGNORECASE)
    pattern3 = re.compile(r"visited positions\s*.*?\s*\[(.*?)\]", re.IGNORECASE)
    # pattern1 = re.compile(r"visited breeze positions.*?\[(.*?)\]", re.IGNORECASE)
    # pattern2 = re.compile(r"visited glitter positions.*?\[(.*?)\]", re.IGNORECASE)
    # pattern3 = re.compile(r"visited positions.*?\[(.*?)\]", re.IGNORECASE)
    matches1 = pattern1.findall(text)
    matches2 = pattern2.findall(text)
    matches3 = pattern3.findall(text)
    if matches1 and matches2 and matches3:
        json_str1 = matches1[-1]
        json_str2 = matches2[-1]
        json_str3 = matches3[-1]
        previous_json = {
            "visited breeze positions": json_str1,
            "visited glitter positions": json_str2,
            "visited positions": json_str3
        }
        return previous_json
    #  第二次提取
    pattern = r'(?i)[\s\S]*?(\{(?:[^{}]|\{[^{}]*\})*\})'
    matches = re.findall(pattern, text, re.DOTALL)
    if matches:
        for match in reversed(matches):
            json_str = match.strip()
            if "visited" in json_str:
                return json_str
    return  None

def extract_guess(text):
    #  第一次提取
    text = text[-600:]
    pattern1 = re.compile(r"pit\s*.*?\s*\[(.*?)\]", re.IGNORECASE)
    matches1 = pattern1.findall(text)
    pattern2 = re.compile(r"gold\s*.*?\s*\[(.*?)\]", re.IGNORECASE)
    matches2 = pattern2.findall(text)
    if matches1 and matches2:
        json_str1 = matches1[-1]
        json_str2 = matches2[-1]
        guess_json = {
            "pit": json_str1,
            "gold": json_str2,
        }
        return guess_json
    #  第二次提取
    pattern = r'[\W_]*guess[\W_]*\s*(.*?)\s*[\W_]*action[\W_]*'
    matches = re.findall(pattern, text, re.IGNORECASE | re.DOTALL)
    if matches:
        result = matches[-1].strip()
        return result
    return None


def generate_scores(log_dir="."):
    """
    遍历指定目录下所有实验日志文件，生成各个实验的最终得分。
    """
    log_files = glob.glob(os.path.join(log_dir, "experiment_log_*.json"))
    scores = []
    for log_file in log_files:
        with open(log_file, "r", encoding="utf-8") as f:
            data = json.load(f)
            scores.append((data.get("log_id"), data.get("final_score")))
    print("Experiment Scores:")
    for log_id, score in sorted(scores):
        print(f"Log {log_id}: Score = {score/100 if score is not None else 'N/A'}")
    scores = [(a, b / 100) for a, b in scores if b is not None]
    return scores

def replay_experiment(env_config=None,log_dir=None,log_filename=None,log_index=None, seed=77):
    """
    根据指定的实验日志文件回放实验过程，逐帧展示实验状态。
    """
    # curr_dir_path = os.path.dirname(os.path.abspath(__file__))
    curr_dir_path = "/home/ubuntu/rlhf/vllm"
    log_filenames = os.listdir(os.path.join(curr_dir_path, 'log', log_dir))

    if isinstance(log_index, int) and -len(log_filenames) <= log_index <= len(log_filenames) - 1:
        print(f"use {log_dir} of index log_filenames[{log_index}], REPLAY THE GAME {log_filenames[log_index]}")
        log_filename = f'{log_filenames[log_index]}'
    if log_filename is None:
        print(f"WARNING, NO log_filename SPECIFIED, REPLAY THE LAST GAME {log_filenames[-1]}")
        log_filename = f'{log_filenames[-1]}'
    if log_filename not in log_filenames:
        print(f"WARNING, CANNOT FIND log_filename {log_filename}, REPLAY THE LAST GAME {log_filenames[-1]}")
        log_filename = f'{log_filenames[-1]}'
    log_filename = f'{curr_dir_path}/log/{log_dir}/{log_filename}'

    with open(log_filename, "r", encoding="utf-8") as f:
        data = json.load(f)
    steps = data.get("steps", [])
    print(f"Replaying experiment log {log_filename}")
    env = CaveEnv(
        env_config=env_config,
        render_mode="human",
        seed=seed
    )
    env.reset()
    env.pit_positions = tuple((x - 1, y - 1) for x, y in eval(data["pit_pos"]))
    env.gold_pos = tuple(i - 1 for i in eval(data["gold_pos"]))

    env.render()
    for step in steps:
        env.render()
        action_str = step['action']
        env.step(action_str)
        print(f"Replaying step {step['step']}: action = {step['action']}, cumulative_reward = {step['cumulative_reward']}")
    env.close()

def visualization_listener(vis_queue, num_processes):
    """
    在主进程中创建一个统一的 Matplotlib 窗口，
    利用 num_processes 个子图分别显示各子进程传递过来的图像，并实时更新。
    每个子图不显示坐标系。
    """
    fig, axes = plt.subplots(1, num_processes, figsize=(5 * num_processes, 5))
    if num_processes == 1:
        axes = [axes]
    plt.ion()
    plt.show()

    # 保存各子进程最新图像的字典
    latest_imgs = {i: None for i in range(num_processes)}
    while True:
        try:
            proc_id, img = vis_queue.get(timeout=0.1)
            latest_imgs[proc_id] = img
        except Empty:
            pass

        for proc_id in range(num_processes):
            if latest_imgs[proc_id] is not None:
                axes[proc_id].clear()
                axes[proc_id].imshow(latest_imgs[proc_id])
                axes[proc_id].set_title(f"Process {proc_id}")
                # 隐藏坐标轴
                axes[proc_id].axis('off')
        plt.pause(0.001)