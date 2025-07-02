# Copyright 2024 Bytedance Ltd. and/or its affiliates
# Copyright 2023 The vLLM team.
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# Adapted from https://github.com/vllm-project/vllm/tree/main/vllm/model_executor/model_loader

from typing import Dict

import torch.nn as nn
from vllm.model_executor.model_loader.utils import set_default_torch_dtype


def update_hf_weight_loader():
    print("no hf weight loader need to be updated")
    return


def load_hf_weights(actor_weights: Dict, vllm_model: nn.Module):
    assert isinstance(actor_weights, Dict)

    #  检查是否是 lora 格式
    actor_weights = convert_lora_weights_to_original(actor_weights)

    with set_default_torch_dtype(next(vllm_model.parameters()).dtype):  # TODO
        if vllm_model.config.tie_word_embeddings and "lm_head.weight" in actor_weights.keys():
            del actor_weights["lm_head.weight"]
        vllm_model.load_weights(actor_weights.items())
    for _, module in vllm_model.named_modules():
        quant_method = getattr(module, "quant_method", None)
        if quant_method is not None:
            quant_method.process_weights_after_loading(module)
        # FIXME: Remove this after Mixtral is updated
        # to use quant_method.
        if hasattr(module, "process_weights_after_loading"):
            module.process_weights_after_loading()
    vllm_model = vllm_model.cuda()


def convert_lora_weights_to_original(actor_weights):
    """
    Convert a LoRA-based weights dictionary to original full weights and rename keys to match the base model.

    Args:
        actor_weights (dict): Dictionary of weights with LoRA structure.

    Returns:
        dict: Dictionary with original weights and cleaned-up keys.
    """
    import torch
    original_weights = {}

    for key in actor_weights.keys():
        # 只处理 base_layer.weight 或 base_layer.bias 的 key
        if 'base_layer.weight' in key or 'base_layer.bias' in key:
            # 构造 A 和 B 的键
            lora_A_key = key.replace('base_layer', 'lora_A.default')
            lora_B_key = key.replace('base_layer', 'lora_B.default')

            # 提取原始值
            base = actor_weights[key]
            A = actor_weights.get(lora_A_key)
            B = actor_weights.get(lora_B_key)

            # 如果 A 和 B 都存在，进行合成
            if A is not None and B is not None:
                lora_weight = torch.matmul(B, A)
                full_weight = base + lora_weight
            else:
                full_weight = base

            # 重命名 key，例如：base_model.model.model.layers.0.self_attn.q_proj.base_layer.weight
            clean_key = key.replace('base_model.model.model.', '') \
                           .replace('base_layer.', '')

            original_weights['model.' + clean_key] = full_weight

        # 处理非 LoRA 参数，直接改名
        elif 'lora_A' not in key and 'lora_B' not in key:
            if 'lm_head' in key:
                clean_key = key.replace('base_model.model.', '')
                original_weights[clean_key] = actor_weights[key]
            else:
                clean_key = key.replace('base_model.model.model.', '')
                original_weights['model.' + clean_key] = actor_weights[key]

    return original_weights