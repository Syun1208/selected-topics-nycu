FROM nvcr.io/nvidia/pytorch:23.10-py3

ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update && apt-get install -y --no-install-recommends \
        git \
        libgl1-mesa-glx \
        libglib2.0-0 \
        libsm6 \
        libxrender1 \
        libxext6 \
        ninja-build \
        build-essential \
        wget \
        curl \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /workspace/hw2

COPY . .

RUN pip install --no-build-isolation -e configs/detectron2

RUN pip install --no-build-isolation -e configs/detrex

RUN pip install --no-cache-dir -r requirements.txt

RUN python -c "import detectron2; import detrex; print('detectron2:', detectron2.__version__); print('detrex OK')"

CMD ["bash"]