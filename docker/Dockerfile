FROM pytorch/pytorch:1.1.0-cuda10.0-cudnn7.5-devel

RUN apt-get -y update --fix-missing
RUN apt-get -y install emacs24 nano ffmpeg ncdu

ARG PIP_EXTRA_INDEX_URL

RUN pip install sklearn scikit-image pyyaml #eai-shuriken

# Copy this entire repo into `src`.
RUN mkdir /src
WORKDIR /src
COPY . .
