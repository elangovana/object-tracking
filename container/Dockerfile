ARG REGION=us-west-2
ARG DEVICE=cpu
ARG AWS_PYTORCH_SAGEMAKER_ECR_ACCOUNT=520713654638
FROM ${AWS_PYTORCH_SAGEMAKER_ECR_ACCOUNT}.dkr.ecr.$REGION.amazonaws.com/sagemaker-pytorch:1.1.0-${DEVICE}-py3

# NOTE: Make sure /opt/ml/code is empty,
# Else Sagemaker container framework will NOT download source code from s3 to this directory

COPY ./src/requirements_prod.txt  /tmp

RUN wget https://repo.anaconda.com/archive/Anaconda3-2019.10-MacOSX-x86_64.sh

RUN bash Anaconda-latest-Linux-x86_64.sh

RUN rm Anaconda3-2019.10-MacOSX-x86_64.sh



RUN conda install -c conda-forge pyarrow


RUN pip install -r /tmp/requirements_prod.txt && rm /tmp/requirements_prod.txt




