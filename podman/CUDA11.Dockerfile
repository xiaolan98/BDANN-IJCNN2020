FROM nvidia/cuda:11.4.1-cudnn8-runtime-ubuntu18.04

# install python
RUN apt-get update && apt-get install -y \
	python3-pip \
	python3.8 \
	&& rm -rf /var/lib/apt/lists/*

# add the virtualenv package
RUN pip3 install virtualenv

# create a python 3.8 environment and activate it
ENV VIRTUAL_ENV="/opt/torchenv"
RUN virtualenv -p $(which python3.8) $VIRTUAL_ENV
ENV PATH="$VIRTUAL_ENV/bin:$PATH"

# install the required dependencies
RUN pip install torch==1.10.1+cu113 torchvision==0.11.2+cu113 torchaudio==0.10.1+cu113 -f https://download.pytorch.org/whl/cu113/torch_stable.html
RUN pip install transformers==4.15.0 pandas==1.3.5 tqdm==4.62.3 sentencepiece==0.1.96 scikit-learn==1.0.2
