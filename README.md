[![Build Status](https://travis-ci.org/elangovana/object-tracking.svg?branch=master)](https://travis-ci.org/elangovana/object-tracking)

# Object tracking
Video object tracking

# Datasets
1. MOT 17 dataset [https://motchallenge.net/data/MOT17/](https://motchallenge.net/data/MOT17/)

# Benchmarks

1. Faster RCNN MOT17 detection benchmark [https://motchallenge.net/results/MOT17Det/](https://motchallenge.net/results/MOT17Det/)

    S. Ren, K. He, R. Girshick, J. Sun. Faster R-CNN: Towards Real-Time Object Detection with Region Proposal Networks. In NIPS, 2015

    
|AP	   | MODA |	MODP	| FAF	|TP	     |FP	 |FN	   |Precision  |Recall|
|------|------|---------|-------|--------|-------|---------|-----------|------|
|0.72  | 68.5 |78.0	    | 1.7	|88,601	 |10,081 |	25,963 |89.8	   |77.3  |

    
# Known issues
1. No multi-gpu training support, only makes use of single gpu

2. Because we are using pretrained models, the model size is quite large and the batch size we can fit into GPU memory is just 8 on a P3 instance. So need to implement gradient accumulation

3. Sagemaker trainining makes use of SPOT instances, need to implement checkpointing to resume training when interrupted
    
# Run object detection

1. To run on command line, using the mot17 dataset
    ```bash
    export PYTHONPATH=./src
    python ./src/experiment_train.py --dataset Mot17DetectionFactory --traindir ./tests/data/clips --valdir tests/data/clips --batchsize 8 --commit_id 763b78c085244fa2fe816f48545cdb520e037b51  --epochs 2 --learning_rate 0.0001 --log-level INFO --model FasterRcnnFactory --momentum 0.9 --patience 20 --weight_decay 5e-05
    ```
    
2. To run on SageMaker, see notebook [Sagemaker.ipynb](Sagemaker.ipynb)
