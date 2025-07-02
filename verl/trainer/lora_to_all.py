
# from transformers import AutoModelForCausalLM, AutoConfig
# from peft import PeftModel, PeftConfig
#
# peft_model_id = "/home/run/wh-verl/rlhf-fsdp/checkpoints_local/Qwen2.5-1.5B-Instruct/lora"
#
# # 加载 PEFT 配置
# config = PeftConfig.from_pretrained(peft_model_id)
#
# # 使用 AutoConfig 来加载基础模型的配置
# base_model_path = config.base_model_name_or_path  # 获取基础模型的路径
# base_model_config = AutoConfig.from_pretrained(base_model_path)
#
# # 加载基础模型
# model = AutoModelForCausalLM.from_pretrained(base_model_path, config=base_model_config)
#
# # 从 PEFT 模型加载 LoRA 调整
# model = PeftModel.from_pretrained(model, peft_model_id)
#
# save_path = "/home/ubuntu/my_models_train"
# model.save_pretrained(save_path)  # 保存模型权重和配置文件
# print(f'----------------finished save to {save_path}-----------------')

from transformers import AutoModelForCausalLM, AutoConfig
from peft import PeftModel, PeftConfig

peft_model_id = "/home/run/wh-verl/rlhf-fsdp/checkpoints_local/Qwen2.5-1.5B-Instruct/lora"

model = AutoModelForCausalLM.from_pretrained(peft_model_id)
print(model)
# save_path = "/home/ubuntu/my_models_train"
# model.save_pretrained(save_path)  # 保存模型权重和配置文件
# print(f'----------------finished save to {save_path}-----------------')