PY_ARGS=${@:1}
PORT=${PORT:-29501}

accelerate launch --multi_gpu --mixed_precision fp16 --gpu_ids 0,1,3,4 --num_processes 4 \
train_geodiffusion_detection_finetune_simultaneous.py \
    --pretrained_model_name_or_path pretrained_diffusers/coco-stuff-256x256 \
    --prompt_version v1 --num_bucket_per_side 256 256 --bucket_sincos_embed \
    --foreground_loss_mode constant --foreground_loss_weight 2.0 --foreground_loss_norm \
    --seed 0 --train_batch_size 4 --gradient_accumulation_steps 3 --gradient_checkpointing \
    --mixed_precision fp16 --num_train_epochs 2 --learning_rate 1e-5 --max_grad_norm 1 \
    --lr_scheduler cosine --lr_warmup_steps 3000 \
    --dataset_config_name configs/data/coco_stuff_256x256.py \
    --uncond_prob 0.1 \
    --max_timestep_rewarding 50 \
    --timestep_resample 6 \
    --output_dir "coco_cycle_detection_simultaneous_finetune_256x256_lr1e-5_maxts50_resample6_reward0.1" \
    --save_ckpt_freq 500
    ${PY_ARGS}