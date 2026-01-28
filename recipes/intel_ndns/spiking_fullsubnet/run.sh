#!/usr/bin/env bash

# For multi-GPU evaluation using Hugging Face Accelerate
accelerate launch --multi_gpu \
    --num_processes=4 \
    --gpu_ids 0,1,2,3 \
    --main_process_port 46601 \
    run.py \
    --config_path conf/middle-model__partition-0-32-128-256__grouping-8-32-64__deep-filtering-5-3-1__synops-1e-8.yaml \
    --eval_batch_size 4 \
    --resume_from_checkpoint /home/xhao/proj/spiking-fullsubnet/recipes/intel_ndns/spiking_fullsubnet_v2/exp/middle-model__partition-0-32-128-256__grouping-8-32-64__deep-filtering-5-3-1__synops-1e-8/checkpoints/epoch_0188 \
    --do_eval true 

# For training, please comment out the above command and use the following command:
accelerate launch --multi_gpu \
    --num_processes=4 \
    --gpu_ids 0,1,2,3 \
    --main_process_port 46601 \
    run.py \
    --config_path conf/middle-model__partition-0-32-128-256__grouping-8-32-64__deep-filtering-5-3-1__synops-1e-8.yaml \
    --eval_batch_size 4