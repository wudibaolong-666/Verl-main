set -x

pos_size3_distill="$HOME/verl_datasets/sft/cot/size3_pos_distill.parquet"
pos_size4_distill="$HOME/verl_datasets/sft/cot/size4_pos_distill.parquet"

torchrun --standalone --nnodes=1 --nproc_per_node=4 \
     -m verl.trainer.fsdp_sft_trainer \
    data.train_files=["$pos_size3_distill","$pos_size4_distill"] \
    data.val_files= \
    data.max_length=16000 \
    data.prompt_key=prompt \
    data.response_key=response \
    data.train_batch_size=16 \
    data.micro_batch_size_per_gpu=4 \
    model.partial_pretrain=/home/ubuntu/my_models/Qwen2.5-1.5B-Instruct \
    model.enable_gradient_checkpointing=False \
    ulysses_sequence_parallel_size=4 \
    use_remove_padding=True \
    optim.lr=1e-5  \
    optim.betas='[0.9, 0.95]' \
    optim.weight_decay=0.01 \
    optim.warmup_steps_ratio=0.1 \
    optim.clip_grad=1.0 \
    trainer.default_local_dir=/home/ubuntu/my_models_train/sft/cot_pos_distill \
    trainer.default_hdfs_dir=null \
    trainer.project_name=qwen2.5_1.5b_sft \
    trainer.experiment_name=cot_pos_distill \
    trainer.total_epochs=12 \
    trainer.seed=1 \
    trainer.logger=['console','swanlab'] $@