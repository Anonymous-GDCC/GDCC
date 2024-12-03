CKPT_PATH=$1
PY_ARGS=${@:2}

python test_geodiffusion.py $CKPT_PATH \
    --dataset_config_name configs/data/coco_stuff_256x256.py \
    --prompt_version v1 --num_bucket_per_side 256 256 \
    --num_inference_steps 50 \
    --nsamples 1 \
    --cfg_scale 4.0 \
    ${PY_ARGS}