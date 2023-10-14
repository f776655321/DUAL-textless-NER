python train_sa_qa.py \
  --model_name_or_path Splend1dchan/byt5lephone_g2p_v1-1024-NMSQA \
  --dataloader_num_workers 4 \
  --do_train \
  --per_device_train_batch_size 16 \
  --gradient_accumulation_step 1 \
  --learning_rate 2e-5 \
  --num_train_epochs 10 \
  --warmup_steps 500 \
  --logging_steps 100 \
  --save_steps 1000 \
  --max_seq_length 1024 \
  --doc_stride 256 \
  --save_strategy "steps" \
  --output_dir SA  \
  --fp16