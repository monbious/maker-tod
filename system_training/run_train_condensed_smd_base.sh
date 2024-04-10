#!/bin/bash

export CUDA_VISIBLE_DEVICES=0
ES=60000
DATA=RRG_qtod_data0_times_gtdb_gesa_times-cr
RMN=others/models/RRG/retriever_camrest_cdnet_data0_times_gtdb_gesa_times-cr_retrieve1_seed-111_ep-15_lr-5e-5_wd-0.01_maxlen-128_bs-108_ngpu-_pln-128_tmp-0.05_hnw-0
python train.py \
    --name new_joint \
    --model_size base \
    --checkpoint_dir others/result/smd/RRG_v3_micro_qtod \
    --retriever_model_name ${RMN} \
    --dataset_name smd \
    --metric_version new1 \
    --train_data others/data/smd/data_used/${DATA}/train.json \
    --eval_data others/data/smd/data_used/${DATA}/val.json \
    --test_data others/data/smd/data_used/${DATA}/test.json \
    --dbs others/data/smd/data_used/${DATA}/all_db.json \
    --per_gpu_eval_batch_size 8 \
    --per_gpu_batch_size 2 \
    --generator_db_maxlength 200 \
    --ranker_db_maxlength 200 \
    --answer_maxlength 128 \
    --top_k_dbs 6 \
    --use_dk True \
    --use_checkpoint \
    --total_steps ${ES} \
    --retriever_total_steps ${ES} \
    --ranker_total_steps ${ES} \
    --end_eval_step ${ES} \
    --use_gt_dbs True \
    --use_retriever_for_gt True \
    --neg_ent_slide_size 3 \
    --generator_distill_retriever True \
    --generator_distill_retriever_start_step 40000 \
    "$@"