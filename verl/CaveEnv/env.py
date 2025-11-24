import time
import gymnasium as gym
from gymnasium import spaces
import random
import numpy as np
import pickle
import json
import os
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import re
from PIL import Image
import pylab
import math

os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'


class CaveEnv(gym.Env):
    metadata = {"render_modes": ["human"], "render_fps": 5}

    def __init__(self, env_config=None, render_mode=None):
        if env_config is None:
            env_config = {}
        # 保存配置
        self.env_config = env_config
        self.done = False

        self.size = env_config.get("size", 4)
        self.env_id = env_config.get("env_id", None)
        self.pit_prob = env_config.get("pit_prob", 0.2)
        self.max_steps = env_config.get("max_steps", 50)
        self.num_pits = env_config.get("num_pits", 2)      #  陷阱
        self.num_golds = env_config.get("num_gold", 1)
        self.start_pos = env_config.get("start_pos", [0, 0])
        self.history_pos = {
            "visited breeze positions": [],
            "visited glitter positions": [],
            "visited positions": []
        }

        # 观测 / 动作空间
        # self.observation_space = spaces.MultiBinary(5)  # [Stench, Breeze, Glitter, Bump, Scream]
        self.observation_space = spaces.MultiBinary(3)  # [Breeze, Glitter, Bump]
        self.action_space = spaces.Discrete(4)

        # 渲染
        self.render_mode = render_mode
        self.window = None
        self.clock = None
        self.cell_size = 60
        self.window_size = self.size * self.cell_size

        # 随机种子
        self.seedConst = env_config.get("seed", 777)
        self.seed(self.seedConst)

        # 初始化内部状态
        self._reset_state()

        self.action_mapping = {
            "<MoveUp>": 0,
            "<MoveDown>": 1,
            "<MoveLeft>": 2,
            "<MoveRight>": 3,
        }

        # 加载可视化图片
        # curr_dir_path = os.path.dirname(os.path.abspath(__file__))
        # self.bg_img = plt.imread(f'{curr_dir_path}/image/background.png')
        # self.wumpus_img = plt.imread(f'{curr_dir_path}/image/wumpus.png')
        # self.wumpus_dead = plt.imread(f'{curr_dir_path}/image/wumpus_dead.png')
        # self.pit_img = plt.imread(f'{curr_dir_path}/image/pit.png')
        # self.gold_img = plt.imread(f'{curr_dir_path}/image/gold.png')
        # self.agent_img = plt.imread(f'{curr_dir_path}/image/agent.png')

    # --------------------------------------------------
    # 工具函数 / 初始化
    # --------------------------------------------------
    def seed(self, seed=777):
        self.np_random, seed = gym.utils.seeding.np_random(seed)
        random.seed(seed)
        np.random.seed(seed)
        return [seed]

    def _reset_state(self):
        """重置环境内部状态变量，并生成 pit/Wumpus 等"""
        self.step_count = 0
        self.agent_pos = self.start_pos
        self.history_pos["visited breeze positions"] = []
        self.history_pos["visited glitter positions"] = []
        if (self.agent_pos[0] + 1, self.agent_pos[1] + 1) not in self.history_pos["visited positions"]:
            self.history_pos["visited positions"].append((self.agent_pos[0] + 1, self.agent_pos[1] + 1))

        # 记录已访问格子（0 基坐标）
        self.visited_positions = {tuple(self.agent_pos)}

        # -------- 生成禁止区域（起始位置及其相邻位置）--------
        forbidden_positions = {tuple(self.agent_pos)}
        directions = [(0, 1), (0, -1), (1, 0), (-1, 0)]  # 上下左右
        for dx, dy in directions:
            nx, ny = self.agent_pos[0] + dx, self.agent_pos[1] + dy
            if 0 <= nx < self.size and 0 <= ny < self.size:
                forbidden_positions.add((nx, ny))

        # -------- 生成金子 --------
        while True:
            gx, gy = random.randrange(self.size), random.randrange(self.size)
            if (gx, gy) not in forbidden_positions:
                break
        self.gold_pos = (gx, gy)
        forbidden_positions.add(self.gold_pos)  # 将金子位置加入禁止区域

        # -------- 生成 pit 确保两者之间欧式距离>2 --------
        candidates_pit = [
            (x, y)
            for x in range(self.size)
            for y in range(self.size)
            if (x, y) not in forbidden_positions
        ]
        default_num = int((self.size * self.size - 2) * self.pit_prob) if self.num_pits is None else self.num_pits
        selected_pits = []
        while len(selected_pits) != default_num:
            selected_pits = []
            for pos in random.sample(candidates_pit, len(candidates_pit)):
                if all(self.dist(pos, existing) >= 3 for existing in selected_pits):
                    selected_pits.append(pos)
                    if len(selected_pits) == default_num:
                        break
        self.pit_positions = set(selected_pits)

        # 其他状态
        self.gold_collected = False
        self.gold_collected_ppo =False
        self.ActionSequence = ""


    def reset(self, options=None):
        """重置环境并返回初始观测"""
        self.seed(self.seedConst)
        super().reset(seed=self.seedConst)
        self._reset_state()

        # 初始观测
        self.CurrentObservation = {
            "grid-based cave size": self.size,
            # "num of gold": self.num_golds,
            "num of pit": self.num_pits,
            "Start position(x, y)": (self.start_pos[0] + 1, self.start_pos[1] + 1),
            "Current position(x, y)":  (self.start_pos[0]+1, self.start_pos[1]+1),
            "previous action": "This is the first step so there is no previous action.",
            "gold_collected": "No",
            "breeze": "No",
            "glitter": "No",
            "bump": "No"
        }

        breeze = self._adjacent_to_any(self.agent_pos, self.pit_positions)
        glitter = self._adjacent_to(self.agent_pos, self.gold_pos) and not self.gold_collected
        if breeze:
            self.CurrentObservation["breeze"] = 'Yes'

        if glitter:
            self.CurrentObservation["glitter"] = "Yes"

        observation = self._get_observation(breeze, glitter, bump=False)

        return observation, {}

        #  for ppo train   2 + 1 + 3 = 6
        # agent_x = self.agent_pos[0] / self.size
        # agent_y = self.agent_pos[1] / self.size
        # gold_collected = self.gold_collected
        # obs = np.array([agent_x, agent_y, gold_collected], dtype=np.float32)
        # obs = np.concatenate((obs, np.array(observation, dtype=np.float32)))
        # return obs

    def step(self, action_text):
        """环境一步模拟"""
        if action_text in [0,1,2,3]:
            action = action_text
        else:
            assert action_text in self.action_mapping, "无效动作！"
            action = self.action_mapping[action_text]
        self.step_count += 1

        breeze = glitter = False
        bump  = False
        reward = -1.0
        terminated = truncated = False
        info = {}

        x, y = self.agent_pos

        # ----------------- 移动 -----------------
        if action in [0, 1, 2, 3]:
            if action == 0:
                nx, ny = x, y + 1
            elif action == 1:
                nx, ny = x, y - 1
            elif action == 2:
                nx, ny = x - 1, y
            else:
                nx, ny = x + 1, y

            if nx < 0 or nx >= self.size or ny < 0 or ny >= self.size:
                bump, nx, ny = True, x, y
            else:
                self.agent_pos = [nx, ny]
                self.visited_positions.add((nx, ny))
                if tuple(self.agent_pos) == self.gold_pos and not self.gold_collected:
                    self.gold_collected, reward = True, reward + 50.0
                    self.CurrentObservation["gold_collected"] = "Yes"
                if self.gold_collected and tuple(self.agent_pos) == tuple(self.start_pos):
                    reward, terminated = reward + 50.0, True
                    info["success"] = True

        # ----------------- 陷阱检查-----------------
        if not terminated:
            if tuple(self.agent_pos) in self.pit_positions:
                terminated, reward, info["dead"] = True, reward - 20.0, "pit"

        # 步数截断
        truncated = self.step_count >= self.max_steps

        if terminated or truncated:
            self.done = True

        # ----------- 更新 CurrentObservation -----------
        x, y = self.agent_pos
        self.CurrentObservation["previous action"] = action_text
        self.CurrentObservation["Current position(x, y)"] = (self.agent_pos[0]+1, self.agent_pos[1]+1)


        if bump:
            self.CurrentObservation["bump"] = "Yes"
        else:
            self.CurrentObservation["bump"] = "No"

        breeze = self._adjacent_to_any(self.agent_pos, self.pit_positions)
        if breeze:
            self.CurrentObservation["breeze"] = "Yes"
        else:
            self.CurrentObservation["breeze"] = "No"

        glitter = self._adjacent_to(self.agent_pos, self.gold_pos) and not self.gold_collected
        if glitter:
            self.CurrentObservation["glitter"] = "Yes"
        else:
            self.CurrentObservation["glitter"] = "No"

        observation = self._get_observation(breeze, glitter, bump)

        #  update previous pos
        if breeze:
            if (self.agent_pos[0] + 1, self.agent_pos[1] + 1) not in self.history_pos["visited breeze positions"]:
                self.history_pos["visited breeze positions"].append((self.agent_pos[0] + 1, self.agent_pos[1] + 1))
        if glitter:
            if (self.agent_pos[0] + 1, self.agent_pos[1] + 1) not in self.history_pos["visited glitter positions"]:
                self.history_pos["visited glitter positions"].append((self.agent_pos[0] + 1, self.agent_pos[1] + 1))
        if (self.agent_pos[0] + 1, self.agent_pos[1] + 1) not in self.history_pos["visited positions"]:
            self.history_pos["visited positions"].append((self.agent_pos[0] + 1, self.agent_pos[1] + 1))

        return observation, reward, terminated, truncated, info

        #  for ppo train   2 + 1 + 3 = 6
        # agent_x = self.agent_pos[0] / self.size
        # agent_y = self.agent_pos[1] / self.size
        # gold_collected = self.gold_collected
        # obs = np.array([agent_x, agent_y, gold_collected], dtype=np.float32)
        # obs = np.concatenate((obs, np.array(observation, dtype=np.float32)))
        # r = 0
        # if tuple(self.agent_pos) == self.gold_pos and not self.gold_collected_ppo:
        #     self.gold_collected_ppo = True
        #     r += 10
        # if self.gold_collected and tuple(self.agent_pos) == tuple(self.start_pos):
        #     r += 20
        # if bump:
        #     r += -2
        # else:
        #     r += -1
        # if tuple(self.agent_pos) in self.pit_positions:
        #     r += -5
        # return obs, r, terminated, truncated, info

    def _adjacent_to(self, pos, target):
        return (abs(pos[0] - target[0]) == 1 and pos[1] == target[1]) or (
            abs(pos[1] - target[1]) == 1 and pos[0] == target[0]
        )

    def _adjacent_to_any(self, pos, targets):
        return any(self._adjacent_to(pos, t) for t in targets)

    def _get_observation(self, breeze, glitter, bump):
        return [int(breeze), int(glitter), int(bump)]

    def render(self, mode='human'):
        if not hasattr(self, 'fig') or self.fig is None:
            if mode == 'rgb_array':
                plt.ioff()
            else:
                plt.ion()
            self.fig, self.ax = plt.subplots()
            self.ax.set_aspect('equal')

        self.ax.clear()
        self.ax.set_xlim(0, self.size)
        self.ax.set_ylim(0, self.size)
        self.ax.set_xticks(range(self.size + 1))
        self.ax.set_yticks(range(self.size + 1))
        self.ax.grid(True)

        self.ax.imshow(self.bg_img,
                       extent=(0, self.size, 0, self.size),
                       origin='upper')

        def show(img, i, j):
            self.ax.imshow(img,
                           extent=(i, i + 1, j, j + 1),
                           origin='upper')

        #  draw pit
        for i, j in self.pit_positions:
            show(self.pit_img, i, j)

        #  draw gold
        if not self.gold_collected:
            i, j = self.gold_pos
            show(self.gold_img, i, j)

        #  draw agent
        i, j = self.agent_pos
        show(self.agent_img, i, j)

        #  draw line of grid
        for i in range(self.size):
            for j in range(self.size):
                rect = patches.Rectangle((i, j),
                                         1, 1,
                                         fill=False,
                                         edgecolor='black',
                                         linewidth=1)
                self.ax.add_patch(rect)

        self.fig.canvas.draw()
        self.fig.canvas.flush_events()

        if mode == 'human':
            plt.pause(1)
            return None
        elif mode == 'rgb_array':
            img = np.frombuffer(self.fig.canvas.tostring_rgb(), dtype=np.uint8)
            img = img.reshape(self.fig.canvas.get_width_height()[::-1] + (3,))
            plt.close(self.fig)
            self.fig = None
            return img
        else:
            return None

    def close(self):
        if self.window is not None:
            self.window = None
            self.clock = None

    def dist(self, a, b):
        return math.sqrt((a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2)

    def get_obs_output_llm(self):
        pro = "The input information are as follows:"
        temp = self.CurrentObservation.copy()

        return pro, temp

