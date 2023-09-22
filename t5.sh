# Change max_seq_len according to model
python train_t5.py \
  --model_name_or_path Splend1dchan/byt5lephone_g2p_v1-1024-NMSQA \
  --dataloader_num_workers 4 \
  --do_train \
  --per_device_train_batch_size 8 \
  --gradient_accumulation_step 4 \
  --learning_rate 2e-5 \
  --num_train_epochs 4 \
  --warmup_steps 500 \
  --logging_steps 100 \
  --save_steps 1000 \
  --max_seq_length 1024 \
  --doc_stride 256 \
  --save_strategy "steps" \
  --output_dir QADAC-t5  \
  --fp16


python train_t5.py \
  --model_name_or_path Splend1dchan/byt5lephone_g2p_v1-1024-NMSQA \
  --dataloader_num_workers 4 \
  --do_train \
  --do_eval \
  --per_device_train_batch_size 8 \
  --per_device_eval_batch_size 8 \
  --gradient_accumulation_step 4 \
  --learning_rate 3e-5 \
  --num_train_epochs 9 \
  --warmup_steps 500 \
  --logging_steps 100 \
  --eval_steps 2 \
  --save_steps 3 \
  --max_seq_length 1024 \
  --doc_stride 256 \
  --save_strategy "steps" \
  --evaluation_strategy "steps" \
  --output_dir QADAC-t5 \
  --fp16