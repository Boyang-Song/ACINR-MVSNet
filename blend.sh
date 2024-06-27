#!/bin/bash
source env.sh
#######train-for val&test#####
data=$(date +"%m%d-%H%M")
CUDA_VISIBLE_DEVICES=0 PYTHONHASHSEED=0 python train_blend.py  --dataset=dtu_yao_blend_caspl --trainpath=$BLEND_TRAINING \
        --trainlist=/home/sby/data/MVS_Data/dataset_low_res/training_list.txt \
        --vallist=/home/sby/data/MVS_Data/dataset_low_res/validation_list.txt \
        --testlist=/home/sby/data/MVS_Data/dataset_low_res/validation_list.txt \
        --batch_size=2 --epochs=16 --lr=0.00025 --wd=0.001 \
        --view_num=3 --inverse_depth=False --numdepth=48 \
        --max_h=576 --max_w=768 \
        --logdir=./checkpoints/$data
#######self-val#####
ckpt=./checkpoints/${data}/model_000016.ckpt
CUDA_VISIBLE_DEVICES=0 python val_blend.py --dataset=dtu_yao_blend_caspl --testpath=$BLEND_TRAINING \
        --testlist=/home/sby/data/MVS_Data/dataset_low_res/validation_list.txt \
        --batch_size=1 --max_h=576 --max_w=768 \
        --view_num=3 --inverse_depth=False --numdepth=48 \
        --loadckpt=$ckpt --outdir=./val_blend

######self-test&fusion#####
ckpt=./checkpoints/${data}/model_000016.ckpt
CUDA_VISIBLE_DEVICES=0 python eval_blend.py --dataset=data_eval_transform_blend_caspl --testpath=$BLEND_TRAINING \
        --testlist=/home/sby/data/MVS_Data/dataset_low_res/validation_list.txt \
        --batch_size=1 --max_h=576 --max_w=768 \
        --view_num=5 --inverse_depth=False --numdepth=96 \
        --loadckpt=$ckpt --outdir=./outputs_blend --isTest=True

testlist=/home/sby/data/MVS_Data/dataset_low_res/validation_list.txt
outdir=./outputs_blend/${data}_model_000016.ckpt
test_dataset=blend

python blendfusion.py --testpath=$BLEND_TRAINING \
                     --testlist=$testlist \
                     --outdir=$outdir \
                     --test_dataset=$test_dataset

#########pretrain-val#####
#ckpt=./checkpoints/pretrained/Bmodel.ckpt
#CUDA_VISIBLE_DEVICES=0 python val_blend.py --dataset=dtu_yao_blend_caspl --testpath=$BLEND_TRAINING \
#        --testlist=/home/sby/data/MVS_Data/dataset_low_res/validation_list.txt \
#        --batch_size=1 --max_h=576 --max_w=768 \
#        --view_num=3 --inverse_depth=False --numdepth=48 \
#        --loadckpt=$ckpt --outdir=./val_blend

#######pretrain-test&fusion#####
#ckpt=./checkpoints/pretrained/Bmodel.ckpt
#CUDA_VISIBLE_DEVICES=0 python eval_blend.py --dataset=data_eval_transform_blend_caspl --testpath=$BLEND_TRAINING \
#        --testlist=/home/sby/data/MVS_Data/dataset_low_res/validation_list.txt \
#        --batch_size=1 --max_h=576 --max_w=768 \
#        --view_num=5 --inverse_depth=False --numdepth=96 \
#        --loadckpt=$ckpt --outdir=./outputs_blend --isTest=True

#testlist=/home/sby/data/MVS_Data/dataset_low_res/validation_list.txt
#outdir=./outputs_blend/pretrained_Bmodel.ckpt
#test_dataset=blend
#
#python blendfusion.py --testpath=$BLEND_TRAINING \
#                     --testlist=$testlist \
#                     --outdir=$outdir \
#                     --test_dataset=$test_dataset

