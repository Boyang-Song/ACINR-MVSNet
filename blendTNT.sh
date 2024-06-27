#!/bin/bash
source env.sh
#######train-for TNT#####
ckptdtu=./checkpoints/pretrained/DmodelTNT.ckpt
data=$(date +"%m%d-%H%M")
CUDA_VISIBLE_DEVICES=0 PYTHONHASHSEED=0 python train_blend.py  --dataset=dtu_yao_blend_caspl --trainpath=$BLEND_TRAINING \
        --trainlist=/home/sby/data/MVS_Data/dataset_low_res/training_list.txt \
        --vallist=/home/sby/data/MVS_Data/dataset_low_res/validation_list.txt \
        --testlist=/home/sby/data/MVS_Data/dataset_low_res/validation_list.txt \
        --batch_size=4 --epochs=16 --lr=0.0002 --wd=0.0001 \
        --view_num=3 --inverse_depth=False --numdepth=48 \
        --max_h=576 --max_w=768 \
        --loadckpt=$ckptdtu --logdir=./checkpoints/$data --forTNT=True

#######self-val#####
ckpt=./checkpoints/${data}/model_000016.ckpt
CUDA_VISIBLE_DEVICES=0 python val_blend.py --dataset=dtu_yao_blend_caspl --testpath=$BLEND_TRAINING \
        --testlist=/home/sby/data/MVS_Data/dataset_low_res/validation_list.txt \
        --batch_size=1 --max_h=576 --max_w=768 \
        --view_num=3 --inverse_depth=False --numdepth=48 \
        --loadckpt=$ckpt --outdir=./val_blend --forTNT=True

#########pretrain-val#####
#ckpt=./checkpoints/pretrained/BmodelTNT.ckpt
#CUDA_VISIBLE_DEVICES=0 python val_blend.py --dataset=dtu_yao_blend_caspl --testpath=$BLEND_TRAINING \
#        --testlist=/home/sby/data/MVS_Data/dataset_low_res/validation_list.txt \
#        --batch_size=1 --max_h=576 --max_w=768 \
#        --view_num=3 --inverse_depth=False --numdepth=48 \
#        --loadckpt=$ckpt --outdir=./val_blend --forTNT=True


