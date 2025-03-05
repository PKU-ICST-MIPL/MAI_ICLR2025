# DDP
# export NCCL_P2P_LEVEL=NVL

python -m torch.distributed.launch --nproc_per_node=8 --master_port 29505 src/train_ddp.py

# python -m torch.distributed.launch --nproc_per_node=8 --master_port 29505 src/clip_finetune_ddp.py
