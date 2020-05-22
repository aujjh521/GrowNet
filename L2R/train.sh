#!/bin/bash


### Feature Table ###
# a9a 123
# ijcnn1 22
# covtype 54
# mnist28 752
# real-sim 20958
# criteo 45
# yahoo 519 - 1
# microsoft 136
dataset=microsoft

BASEDIR=$(dirname "$0")
OUTDIR="${BASEDIR}/ckpt/"

if [ ! -d "${OUTDIR}" ]
then   
    echo "Output dir ${OUTDIR} does not exist, creating..."
    mkdir -p ${OUTDIR}
fi    

CUDA_VISIBLE_DEVICES=0 python -u main_l2r_pairwise_cv_experiment.py \
    --data_dir ${BASEDIR}/../data \
    --model_version main_l2r_pairwise_cv_experiment.py \
    --model_order second \
    --feat_d 136 \
    --hidden_d 128 \
    --boost_rate 1 \
    --lr 0.005 \
    --L2 1.0e-3 \
    --num_nets 40 \
    --data ${dataset} \
    --batch_size 10000 \
    --epochs_per_stage 2 \
    --correct_epoch 2 \
    --normalization True \
    --sigma 1. \
    --cv True \
    --cuda