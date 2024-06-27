#!/bin/bash
source env.sh
######train-for val&test#####
data=$(date +"%m%d-%H%M")
CUDA_VISIBLE_DEVICES=0 PYTHONHASHSEED=0 python train.py --dataset=dtu_yao --trainpath=$MVS_TRAINING \
        --trainlist=lists/dtu/train.txt --vallist=lists/dtu/val.txt --testlist=lists/dtu/test.txt \
        --batch_size=2 --epochs=16 --lr=0.00025 --wd=0.001 \
        --view_num=3 --inverse_depth=True --numdepth=48 --interval_scale=4.24 \
        --max_h=512 --max_w=640 \
        --logdir=./checkpoints/$data

####self-test&fusion#####
ckpt=./checkpoints/${data}/model_000016.ckpt
CUDA_VISIBLE_DEVICES=0 python eval.py --dataset=data_eval_transform --testpath=$DTU_TESTING \
        --testlist=lists/dtu/test.txt \
        --batch_size=1 --max_h=960 --max_w=1280 \
        --view_num=5 --inverse_depth=True --numdepth=96 --interval_scale=2.13 \
        --loadckpt=$ckpt --outdir=./outputs_dtu --isTest=True

testlist=lists/dtu/test.txt
outdir=./outputs_dtu/${data}_model_000016.ckpt
test_dataset=dtu
python dtufusion.py --testpath=$DTU_TESTING \
                     --testlist=$testlist \
                     --outdir=$outdir \
                     --max_h=960 --max_w=1280 --base_image_size=64 \
                     --test_dataset=$test_dataset


#######pretrain-test&fusion#####
#ckpt=./checkpoints/pretrained/Dmodel.ckpt
#CUDA_VISIBLE_DEVICES=0 python eval.py --dataset=data_eval_transform --testpath=$DTU_TESTING \
#        --testlist=lists/dtu/test.txt \
#        --batch_size=1 --max_h=1152 --max_w=1600 \
#        --view_num=5 --inverse_depth=True --numdepth=96 --interval_scale=2.13 \
#        --loadckpt=$ckpt --outdir=./outputs_dtu --isTest=True
#
#testlist=lists/dtu/test.txt
#outdir=./outputs_dtu/pretrained_Dmodel.ckpt
#test_dataset=dtu
#python dtufusion.py --testpath=$DTU_TESTING \
#                     --testlist=$testlist \
#                     --outdir=$outdir \
#                     --max_h=1152 --max_w=1600 --base_image_size=64 \
#                     --test_dataset=$test_dataset
