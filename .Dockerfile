
# https://hub.docker.com/r/pytorch/pytorch/tags
# baselines require PyTorch 2.3+
# doesn't require a specific CUDA version so long as it runs on your system
FROM pytorch/pytorch:2.7.0-cuda11.8-cudnn9-runtime

# most dependencies are via Zoobot
ADD requirements.txt /tmp
RUN pip install -r /tmp/requirements.txt

RUN pip install "zoobot[pytorch]"
RUN pip install -e .

# add this folder as a volume
# docker run --gpus all -v $PWD:/tmp -w /tmp -it this_image python gz_evo/classification/train.py 