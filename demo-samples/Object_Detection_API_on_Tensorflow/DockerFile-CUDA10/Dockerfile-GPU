FROM qhub/tfod-ces2019:1.0-gpu as source
FROM tensorflow/tensorflow:1.13.1-gpu-py3-jupyter


MAINTAINER QNAP_TimHsu

# install dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    git \
    protobuf-compiler \
    python3-lxml \
    python3-tk \
    python3-setuptools \
    wget \
    unzip \
	cmake \
	libsm6 \
	libxext6 \
	libavcodec-dev \
	libavformat-dev \
	libswscale-dev

RUN pip install --user Cython \
	contextlib2 \
	pillow \
	gevent \
	gunicorn \
	flask \
	opencv-python && \
	ln -s /root/.local/bin/gunicorn /usr/bin/ && \
	ln -s /root/.local/bin/flask /usr/bin

WORKDIR /notebooks

COPY --from=source /notebooks/ /notebooks/


EXPOSE 8888
EXPOSE 6006

ENV LD_LIBRARY_PATH /usr/local/nvidia/lib/:$LD_LIBRARY_PATH
ENV PATH /usr/local/nvidia/bin:$PATH
ENV PYTHONPATH /notebooks/models/research:/notebooks/models/research/slim:$PYTHONPATH

CMD source /etc/bash.bashrc && jupyter notebook --notebook-dir=/notebooks --ip 0.0.0.0 --no-browser --allow-root

