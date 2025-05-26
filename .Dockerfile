
# https://hub.docker.com/r/pytorch/pytorch/tags
# baselines require PyTorch 2.3+
# doesn't require a specific CUDA version so long as it runs on your system
FROM pytorch/pytorch:2.7.0-cuda11.8-cudnn9-runtime

# most dependencies are via Zoobot
# https://github.com/mwalmsley/zoobot/blob/main/setup.py#L63 pytorch-gpu adds pytorch etc (skipped here as installed already)
# https://github.com/mwalmsley/zoobot/blob/main/setup.py#L104 core includes galaxy-datasets which adds a few more dependencies

# additional dependencies for zoobot-foundation


ADD requirements.txt /tmp
RUN pip install -r /tmp/requirements.txt

RUN pip install "zoobot[pytorch-cu121]" --extra-index-url https://download.pytorch.org/whl/cu121


# add this folder as a volume
# docker run --gpus all -v $PWD:/tmp -w /tmp -it this_image python train.py