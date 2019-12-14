[![Build Status](https://travis-ci.org/elangovana/object-tracking.svg?branch=master)](https://travis-ci.org/elangovana/object-tracking)

# Object tracking
Video object tracking

# Datasets
1. MOT 17 dataset [https://motchallenge.net/data/MOT17/](https://motchallenge.net/data/MOT17/)


# Run object detection

1. To run on command line, using the mot17 dataset
    ```bash
    export PYTHONPATH=./src
    python ./src/experiment_train.py --dataset Mot17DetectionFactory --traindir ./tests/data/clips --valdir tests/data/clips --batchsize 8 --commit_id 763b78c085244fa2fe816f48545cdb520e037b51  --epochs 2 --learning_rate 0.0001 --log-level INFO --model FasterRcnnFactory --momentum 0.9 --patience 20 --weight_decay 5e-05
    ```
    
2. To run on SageMaker, see notebook [Sagemaker.ipynb](Sagemaker.ipynb)
