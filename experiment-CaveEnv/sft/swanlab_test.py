import swanlab
import random
import os

SWANLAB_API_KEY = os.environ.get("SWANLAB_API_KEY", "EToXAtdrjIz6vAeW9d9VE")
SWANLAB_LOG_DIR = os.environ.get("SWANLAB_LOG_DIR", "/home/run/wh/logs/swanlab")
SWANLAB_MODE = os.environ.get("SWANLAB_MODE", "cloud")
if SWANLAB_API_KEY:
    swanlab.login(SWANLAB_API_KEY)  # NOTE: previous login information will be overwritten
# 创建一个SwanLab项目
swanlab.init(
    # 设置项目名
    project="my-awesome-project",

    # 设置超参数
    config={
        "learning_rate": 0.02,
        "architecture": "CNN",
        "dataset": "CIFAR-100",
        "epochs": 10
    },
    logdir=SWANLAB_LOG_DIR,
    mode=SWANLAB_MODE
)

# 模拟一次训练
epochs = 10
offset = random.random() / 5
for epoch in range(2, epochs):
    acc = 1 - 2 ** -epoch - random.random() / epoch - offset
    loss = 2 ** -epoch + random.random() / epoch + offset

    # 记录训练指标
    swanlab.log({"acc": acc, "loss": loss})

# [可选] 完成训练，这在notebook环境中是必要的
swanlab.finish()