CKPT_PATH=${1:-"pretrained_diffusers/GeoDiffusion_512x512_GDCC"}
PY_ARGS=${@:2}

python test_geodiffusion.py $CKPT_PATH \
    --dataset_config_name configs/data/coco_stuff_512x512.py \
    --prompt_version v1 --num_bucket_per_side 256 256 \
    --num_inference_steps 50 \
    --nsamples 1 \
    --cfg_scale 4.5 \
    ${PY_ARGS}