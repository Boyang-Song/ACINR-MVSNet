#!/bin/bash
source env.sh

ckpt=./checkpoints/pretrained/BTNTRe.ckpt
CUDA_VISIBLE_DEVICES=0 python eval_tnt.py --dataset=data_eval_transform_padding --testpath=$TP_TESTING \
        --testlist=lists/tp_list_int_1024.txt \
        --batch_size=1 --max_h=1056 --max_w=2048 \
        --inverse_depth=True --numdepth=192 \
        --loadckpt=$ckpt --outdir=./outputs_tnt

CUDA_VISIBLE_DEVICES=0 python eval_tnt.py --dataset=data_eval_transform_padding --testpath=$TP_TESTING \
        --testlist=lists/tp_list_int_960.txt \
        --batch_size=1 --max_h=1056 --max_w=1920 \
        --inverse_depth=True --numdepth=192 \
        --loadckpt=$ckpt --outdir=./outputs_tnt

testlist=./lists/tp_list_int.txt
outdir=./outputs_tnt/pretrained_BTNTRe.ckpt
test_dataset=tnt
python tntfusion.py --testpath=$TP_TESTING \
                     --testlist=$testlist \
                     --outdir=$outdir \
                     --test_dataset=$test_dataset