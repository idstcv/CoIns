# CoIns 
PyTorch Implementation for Our ICCV'21 Paper: "Weakly Supervised Representation Learning with Coarse Labels"

## Requirements
* Python 3.8
* PyTorch 1.7

## Usage: MoCo-v2 + Coarse Labels

1. CoIns Model Training
```
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
python coins_train.py \
    -j 32 \
    -a resnet50 \
    --lr 0.03 \
    -p 1000 \
    --batch-size 256 \
    --coarse-loss-w 0.5 \
    --mlp --moco-t 0.2 --aug-plus --cos \
    --log coins_e200_coarse_w_0.5 \
    --dist-url 'tcp://localhost:'${RANDOM} --multiprocessing-distributed --world-size 1 --rank 0 \
    [your imagenet-folder with train and val folders] | tee log/coins_e200_coarse_w_0.5.log
```

2. Linear Classification: official implementation from MoCo
```
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
python main_lincls.py \
    -a resnet50 \
    -j 32 \
    --lr 30.0 \
    --batch-size 256 \
    --pretrained [path to your CoIns model] \
    --dist-url 'tcp://localhost:'${RANDOM} --multiprocessing-distributed --world-size 1 --rank 0 \
    [your imagenet-folder with train and val folders]
```

3. Retrieval Evaluation
```
CUDA_VISIBLE_DEVICES=0 \
python coins_eval.py \
    --gpu 0 \
    --pretrained [path to your CoIns model] \
    [your imagenet-folder with val folder]
```

## Model:

CoIns: [Google Drive](https://drive.google.com/file/d/1XK1GkLhLplHMKHs-jF38T8-v-ZWcIkrM/view?usp=sharing)

    
## Citation
If you use the package in your research, please cite our paper:
```
@inproceedings{xu2021wk,
  author    = {Yuanhong Xu and
                Qi Qian and
                Hao Li and
                Rong Jin and
                Juhua Hu},
  title     = {Weakly Supervised Representation Learning with Coarse Labels},
  booktitle = {{IEEE} International Conference on Computer Vision, {ICCV} 2021},
  year      = {2021}
}
```