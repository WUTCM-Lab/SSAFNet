#!/bin/sh
################### Train #################
CUDA_VISIBLE_DEVICES=0
DATE=`date '+%Y%m%d-%H%M%S'`
echo ${DATE}


# single GPU
#python train.py --dataset SL1 --end_epoch 100 \
#    --lr 0.0003 --train_batchsize 2 --models bs --head bs \
#    --crop_size 512 512 --use_mixup 0 --use_edge 1 --information ${MODEL} > log/${DATE}+${MODEL}.log

# dataset: SL1 SLSD
MODEL="new-fuse" # 改--master_port
python -m torch.distributed.launch --nproc_per_node=2 --master_port 29502 \
      train.py --dataset SL1 --end_epoch 100 \
      --lr 0.0003 --train_batchsize 2 --models bs --head bs \
      --crop_size 512 512 --use_mixup 0 --use_edge 1 --information ${MODEL} > log_new/${DATE}+${MODEL}.log









