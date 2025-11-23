#!/usr/bin/env bash
set -x  # 启用调试模式，执行前打印每条命令
GPUS_PER_NODE=2  # 每台机器使用的 GPU 数量
MASTER_ADDR={master_address}":"{port}  # 设置主节点地址和端口（需替换为实际值）
NNODES=1  # 总共参与训练的机器数量
JOB_ID=107  # 本次 rendezvous 会话的唯一 ID
torchrun \  # 启动分布式 PyTorch 训练
    --nproc_per_node $GPUS_PER_NODE \  # 每台机器启动的进程数量（应等于 GPU 数）
    --nnodes $NNODES \  # 指定参与训练的节点总数
    --node_rank 0 \  # 当前节点的编号（主节点为0）
    --rdzv_endpoint $MASTER_ADDR \  # 节点间协调通信的 rendezvous 端点
    --rdzv_id $JOB_ID \  # rendezvous 会话标识
    --rdzv_backend c10d \  # 使用 c10d 作为 rendezvous 后端
    goal_gen/train.py_s2sv2 \  # 训练脚本入口
    --config ${@:1} \  # 传入所有附加参数作为配置文件
    --gpus $GPUS_PER_NODE \  # 指定使用的 GPU 数量
    --num_nodes $NNODES  # 指定节点数量
