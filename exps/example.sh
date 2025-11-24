export MODEL_NAME="runwayml/stable-diffusion-v1-5"
export OUTPUT_DIR="../models/KPO_sensation_aim"  #add output path here
export PICK="../Data/RLHF_DATA_sensation_aim" # add data path here
accelerate launch train_kto_sd_v1.5.py \
  --resolution 512 \
  --pretrained_model_name_or_path=$MODEL_NAME  \
  --train_data_dir=$PICK \
  --pick_split train \
  --output_dir=$OUTPUT_DIR \
  --mixed_precision="fp16" \
  --dataset_name=kashif/pickascore \
  --train_batch_size=2 \
  --gradient_accumulation_steps=1 \
  --gradient_checkpointing \
  --use_8bit_adam \
  --rank=8 \
  --report_to="wandb" \
  --lr_scheduler="constant" \
  --learning_rate=1e-07 --lr_scheduler="constant_with_warmup" --lr_warmup_steps=200 \
  --max_train_steps=20000 \
  --checkpointing_steps=500 \
  --run_validation --validation_steps=50 --validation_prompt='Generate an advertisement image that evokes the weightlessness sensation and conveys the following messages: \n I shoud drink this beer because it is light' \
  --seed="0" \
  --report_to="wandb" \
  --checkpoints_total_limit 4 \
  --policy gt_label_wi_l \
  --dataloader pick \
  --loss kto \
  --halo sigmoid \
  --positive_ratio 0.8 \
  --fixed_noise ${@}

  
