## Installation

#### Requiements
* python3.6
* pytorch 1.6.0
* numpy
* matplotlib
* opencv
* nvidia gpu + cuda cudnn

```
pip install -r requiements.txt

# setup for roi_layers
python setup.py build develop

# fvcore
sudo pip install 'git+https://github.com/facebookresearch/fvcore'

# detectron2
python -m pip install 'git+https://github.com/facebookresearch/detectron2.git'

# Note that, after installed detectron2 to local,
# please add the content of ./extra_panptic.py into class Visualizer of local/detectron2/utils/visualizer.py
```

## Getting Started
### Training
- Train the model by single GPU:
```bash
python train.py --dataset summer2winter --s2w_dir ./datasets/summer2winter_256x256_aug 
```
- Train the model by multiple GPUs (e.g., gpu 0, 1, 2, 3) after DistributedDataParallel training setting and setting dataloader as "distributed way" of get_dataloader in ./data/s2w_custom_mask.py:
```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 python -m torch.distributed.launch train_mgpus.py --dataset summer2winter --s2w_dir ./datasets/summer2winter_256x256_aug
```
The trained models will be saved to: `./checkpoints/result_summer2winter/models/`.

### Testing
- Test the model by single GPU:
```bash
python test.py --dataset summer2winter --s2w_dir ./datasets/summer2winter_256x256_aug 
```
- Test the model by multiple GPUs (e.g., gpu 0, 1, 2, 3) after DistributedDataParallel training setting and setting dataloader as "distributed way" of get_dataloader in ./data/s2w_custom_mask.py:
```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 python -m torch.distributed.launch test_mgpus.py --dataset summer2winter --s2w_dir ./datasets/summer2winter_256x256_aug
```
The tested results will be saved to: `./results/`.

