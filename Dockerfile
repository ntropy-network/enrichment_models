# Main stage
FROM nvidia/cuda:11.6.1-cudnn8-devel-ubuntu20.04 AS cuda_machine
ARG DEBIAN_FRONTEND=noninteractive
ENV PIP_CACHE_DIR=/var/cache/buildkit/pip
ENV TRANSFORMERS_CACHE=/var/cache/buildkit/transformers
ENV BASE_MODEL=decapoda-research/llama-7b-hf
WORKDIR /enrichment_models


# Setup Python 3.10
RUN --mount=type=cache,mode=0755,target=/var/cache/apt \
    apt-get update && \
    apt-get install software-properties-common -y && \
    add-apt-repository ppa:deadsnakes/ppa && \
    apt update && \
    apt install -y python3.10 python3.10-distutils curl && \
    curl https://bootstrap.pypa.io/get-pip.py | python3.10 && \
    python3.10 -m pip install --upgrade pip && \
    apt-get clean

RUN apt install git -y

RUN pip install poetry==1.3.1

# Copy files
COPY . /enrichment_models

RUN poetry export -f requirements.txt --without-hashes --output requirements.txt

RUN --mount=type=cache,mode=0755,target=/var/cache/buildkit/pip \
    pip install -r requirements.txt

# Fix pytorch cublas error
RUN pip install torch==1.13.1+cu116 --extra-index-url https://download.pytorch.org/whl/cu116

ENV PYTHONPATH /enrichment_models

FROM cuda_machine AS cuda_machine-kubeflow
ENV DEBIAN_FRONTEND=noninteractive
ENV TZ=Etc/UTC

RUN apt-get update && apt-get install -y locales htop mc git wget curl tmux screen \
    && curl https://getmic.ro | bash && mv micro /usr/local/bin && chmod +x /usr/local/bin/micro

# make things friendly to kubeflow
ENV NB_USER jovyan
ENV NB_UID 1000
ENV NB_PREFIX /
ENV HOME /home/$NB_USER
ENV SHELL /bin/bash
ENV PYTHONPATH "${PYTHONPATH}:/llama-training/lora_finetuning/"
SHELL ["/bin/bash", "-c"]

# create user and set required ownership
RUN useradd -M -s /bin/bash -N -u ${NB_UID} ${NB_USER} \
    && mkdir -p ${HOME} \
    && chown -R ${NB_USER}:users ${HOME} \
    && chown -R ${NB_USER}:users /usr/local/bin

# install github cli
RUN curl -fsSL https://cli.github.com/packages/githubcli-archive-keyring.gpg | dd of=/usr/share/keyrings/githubcli-archive-keyring.gpg \
    && chmod go+r /usr/share/keyrings/githubcli-archive-keyring.gpg \
    && echo "deb [arch=$(dpkg --print-architecture) signed-by=/usr/share/keyrings/githubcli-archive-keyring.gpg] https://cli.github.com/packages stable main" | tee /etc/apt/sources.list.d/github-cli.list > /dev/null \
    && apt update && apt install gh -y

RUN python3.10 -m pip install jupyterlab pdbpp && ln -s /usr/bin/python3.10 /usr/bin/python && pip install awscli
# Fix
RUN python3.10 -m pip install botocore==1.27.59

# Required by Kubeflow notebook server
EXPOSE 8888
ENTRYPOINT ["sh","-c", "python3.10 -m jupyterlab --notebook-dir=/home/jovyan --ip=0.0.0.0 --no-browser --allow-root --port=8888 --NotebookApp.token='' --NotebookApp.password='' --NotebookApp.allow_origin='*' --NotebookApp.base_url=${NB_PREFIX}"]