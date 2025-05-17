# GDCC

This anonymous repository contains the implementation of the paper:

> Cycle-Consistent Learning for Joint Layout-to-Image Generation and Object Detection <br>

In this paper, we propose a generation-detection cycle consistent (GDCC) learning framework that jointly optimizes both layout-to-image (L2I) generation and object detection (OD) tasks in an end-to-end manner. The key of GDCC lies in the inherent duality between the two tasks, where L2I takes all object boxes and labels as input conditions to generate images, and OD maps images back to these layout conditions. Specifically, in GDCC , L2I generation is guided by a layout translation cycle loss, ensuring that the layouts used to generate images align with those predicted from the synthesized images. Similarly, OD benefits from an image translation cycle loss, which enforces consistency between the synthesized images fed into the detector and those generated from predicted layouts. 
![img](./images/overview.png)



## Installation

Clone this repo and create the GDCC environment with conda. We test the code under `python==3.8.13, pytorch==1.12.1, cuda=11.3` on Tesla V100 GPU servers.

1. Initialize the conda environment:

   ```bash
   conda create -n gdcc python=3.8 -y
   conda activate gdcc
   ```

2. Install the required packages:

   ```bash
   cd gdcc
   # when running training
   pip install -r requirements/train.txt
   # only when running inference with DPM-Solver++
   pip install -r requirements/dev.txt
   ```



## Download Pre-trained L2I Generation Models

We provide original L2I generation model and the model fine-tuned with GDCC for comparison. Download and put them into `./pretrained_diffusers/`.

|        Dataset        |  L2I Model   | GDCC Fine-tune | Image Resolution | Grid Size |                           Download                          |
| :-------------------: |:------------:|:--------------:| :--------------: | :-------: | :----------------------------------------------------------: |
|      COCO-Stuff       | GeoDiffusion |       ×        |     256x256      |  256x256  | [HF Hub](https://huggingface.co/KaiChen1998/geodiffusion-coco-stuff-256x256) |
|      COCO-Stuff       | GeoDiffusion |       √        |     256x256      |  256x256  | [HF Hub](https://huggingface.co/AnonymousGDCC/GeoDiffusion_256x256_GDCC/tree/main) |
|      COCO-Stuff       | GeoDiffusion |       ×        |     512x512      |  256x256  | [HF Hub](https://huggingface.co/KaiChen1998/geodiffusion-coco-stuff-512x512) |
|      COCO-Stuff       | GeoDiffusion |       √        |     512x512      |  256x256  | [HF Hub](https://huggingface.co/AnonymousGDCC/GeoDiffusion_512x512_GDCC/tree/main) |




## Generate images with L2I Generation Models fine-tuning with GDCC

Download the pre-trained models and put them under the root directory. Run the following commands to run detection data generation with GeoDiffusion. For simplicity, we embed the layout definition process in the file `run_layout_to_image.py` directly. Check [here](./run_layout_to_image.py#L75-L82) for detailed definition. Run:

```bash
python run_layout_to_image.py $CKPT_PATH --output_dir ./results/
```

## Evaluate L2I Generation Models
### 1. Generate Images according to COCO annotations
For the 256x256 model:
```bash
bash tools/dist_test.sh
```
For the 512x512 model:
```bash
bash tools/dist_test_512x512.sh
```
**Note:** If you need to compute the **YOLO_Score**, please set `--nsamples` to `5`.
### 2. Evaluate L2I Generation Models based on the generated images
For FID and YOLO Scores, Please refer to [LAMA](https://github.com/ZejianLi/LAMA/tree/main).

## Download Pre-trained Detection Models
We provide original L2I detection model and the model fine-tuned with GDCC for comparison. Download and put them into `./pretrained_detectors/`.
|        Detection Model        |  Backbone   |  Configuration   | Original CKPT | GDCC Fine-tuned CKPT|
| :-------------------: |:------------:|:------------:|:--------------:| :--------------: |
|      Faster-R-CNN     |R50           | [config](https://github.com/open-mmlab/mmdetection/blob/main/configs/faster_rcnn/faster-rcnn_r50_fpn_1x_coco.py) |       [ckpt](https://download.openxlab.org.cn/models/mmdetection/FasterR-CNN/weight/faster-rcnn_r50_fpn_1x_coco)        |     [ckpt](https://huggingface.co/AnonymousGDCC/Faster-R-CNN_r50_GDCC/tree/main)      |
|      Cascade-R-CNN    |R50           | [config](https://github.com/open-mmlab/mmdetection/blob/main/configs/cascade_rcnn/cascade-rcnn_r50_fpn_1x_coco.py) |       [ckpt](https://download.openmmlab.com/mmdetection/v2.0/cascade_rcnn/cascade_rcnn_r50_fpn_1x_coco/cascade_rcnn_r50_fpn_1x_coco_20200316-3dc56deb.pth)        |     [ckpt](https://huggingface.co/AnonymousGDCC/Cascade-R-CNN_r50_GDCC/tree/main)      |
|      DINO             |R50           | [config](https://github.com/open-mmlab/mmdetection/blob/main/configs/dino/dino-4scale_r50_8xb2-12e_coco.py) |       [ckpt](https://download.openmmlab.com/mmdetection/v3.0/dino/dino-4scale_r50_8xb2-12e_coco/dino-4scale_r50_8xb2-12e_coco_20221202_182705-55b2bba2.pth)        |     [ckpt](https://huggingface.co/AnonymousGDCC/DINO_r50_GDCC/tree/main)      | 
|      CO-DETR          |R50           | [config](https://github.com/open-mmlab/mmdetection/blob/main/projects/CO-DETR/configs/codino/co_dino_5scale_r50_lsj_8xb2_1x_coco.py) |       [ckpt](https://download.openmmlab.com/mmdetection/v3.0/codetr/co_dino_5scale_r50_lsj_8xb2_1x_coco/co_dino_5scale_r50_lsj_8xb2_1x_coco-69a72d67.pth)       |     [ckpt](https://huggingface.co/AnonymousGDCC/CO-DETR_r50_GDCC/tree/main)      |

### Evaluate Detection Models
To evaluate detection models, please refer to [MMdetection Instructions](https://github.com/open-mmlab/mmdetection/blob/main/docs/en/user_guides/test.md).

## Train GDCC

### 1. Prepare dataset

We primarily use the [nuImages](https://www.nuscenes.org/nuimages) and [COCO-Stuff](https://cocodataset.org/#home) datasets for training GeoDiffusion. Download the image files from the official websites. For better training performance, we follow [mmdetection3d](https://github.com/open-mmlab/mmdetection3d/blob/main/configs/nuimages/README.md/#introduction) to convert the nuImages dataset into COCO format, while the converted annotation file for COCO-Stuff can be download via [HuggingFace](https://huggingface.co/datasets/KaiChen1998/coco-stuff-geodiffusion). The data structure should be as follows after all files are downloaded.

```
├── data
│   ├── coco
│   │   │── coco_stuff_annotations
│   │   │   │── train
│   │   │   │   │── instances_stuff_train2017.json
│   │   │   │── val
│   │   │   │   │── instances_stuff_val2017.json
│   │   │── train2017
│   │   │── val2017
│   ├── nuimages
│   │   │── annotation
│   │   │   │── train
│   │   │   │   │── nuimages_v1.0-train.json
│   │   │   │── val
│   │   │   │   │── nuimages_v1.0-val.json
│   │   │── samples
```

### 2. Launch distributed training


```bash
# fine-tune L2I generation model
bash tools/dist_train_finetune_generator_coco.sh \
	--dataset_config_name configs/data/coco_stuff_256x256.py \
	--output_dir work_dirs/gdcc_g_geodiffusion_coco_stuff_256x256
	
# fine-tune L2I generation model and detection model
bash tools/dist_train_finetune_generator_detector_coco_alternative.sh \
	--dataset_config_name configs/data/coco_stuff_256x256.py \
	--output_dir work_dirs/gdcc_gd_geodiffusion_coco_stuff_256x256


# fine-tune L2I generation model and detection model simultaneously
bash tools/dist_train_finetune_generator_detector_coco_simultaneous.sh \
	--dataset_config_name configs/data/coco_stuff_256x256.py \
	--output_dir work_dirs/gdcc_gd_simultaneous_geodiffusion_coco_stuff_256x256
```



### 3. Launch batch inference


```bash
# COCO-Stuff
# We encourage readers to check https://github.com/ZejianLi/LAMA?tab=readme-ov-file#testing
# to report quantitative results on COCO-Stuff L2I benchmark.
bash tools/dist_test.sh PATH_TO_CKPT \
	--dataset_config_name configs/data/coco_stuff_256x256.py
```


## Qualitative Results

More results can be found in the main paper.

![img](./images/qualitative_1.PNG)

![img](./images/qualitative_2.PNG)






## Acknowledgement

We adopt the following open-sourced projects:

- [geodiffusion](https://github.com/KaiChen1998/GeoDiffusion): GeoDiffusion for L2I generation.
- [controlnet](https://github.com/lllyasviel/ControlNet): ControlNet for Controllable generation.
- [controlnet++](https://github.com/xinsir6/https://github.com/liming-ai/ControlNet_Plus_Plus): improve controls with consistency feedback.
- [diffusers](https://github.com/huggingface/diffusers/): basic codebase to train Stable Diffusion models.
- [mmdetection](https://github.com/open-mmlab/mmdetection): dataloader to handle images with various geometric conditions.
- [mmdetection3d](https://github.com/open-mmlab/mmdetection3d) & [LAMA](https://github.com/ZejianLi/LAMA): data pre-processing of the training datasets.
