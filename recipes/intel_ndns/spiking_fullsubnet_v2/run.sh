# accelerate launch --multi_gpu --num_processes=4 --gpu_ids 4,5,6,7 --main_process_port 46602 run.py --config_path default.yaml \
#     --output_dir exp/middle \
#     --eval_dataset.limit=200 \
#     --fb_hidden_size 320 \
#     --sb_hidden_size 224 \
#     --df_orders 5 3 1 \
#     --per_device_train_batch_size 64 \
#     --save_epoch_interval 5 \
#     --eval_epoch_interval 5

accelerate launch --multi_gpu --num_processes=4 --gpu_ids 0,1,2,3 --main_process_port 46601 run.py --config_path default.yaml --output_dir exp/small --eval_dataset.limit 200 --fb_hidden_size 320 --sb_hidden_size 224 --df_orders 3 1 1 --per_device_train_batch_size 120 --save_epoch_interval 5 --eval_epoch_interval 5

# Small
accelerate launch --multi_gpu --num_processes=4 --gpu_ids 0,1,2,3 --main_process_port 46601 run.py --config_path default.yaml --output_dir exp/small --eval_dataset.limit 200 --fb_hidden_size 240 --sb_hidden_size 160 --df_orders 3 1 1 --per_device_train_batch_size 120 --save_epoch_interval 5 --eval_epoch_interval 5 --resume_from_checkpoint
