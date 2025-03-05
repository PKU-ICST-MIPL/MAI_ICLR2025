# python src/blip_validate.py \
#    --dataset FashionIQ \
#    --blip-model-name 'blip2_cir_align_prompt' \
#    --model-path '/mnt/longvideo/chenyanzhe/Multiturn/ckpt/sprc_fiq.pt'

# python src/test.py -- model-path '/mnt/longvideo/chenyanzhe/Multiturn/ckpt/'

python -m torch.distributed.launch --nproc_per_node=8 --master_port 29505 src/test.py
