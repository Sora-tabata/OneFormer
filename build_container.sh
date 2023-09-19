docker stop oneformer_v2
docker rm oneformer_v2



docker pull sorajang/oneformer_v2



# Create a new container
docker run -i -t --gpus all -d \
--shm-size=12gb \
-v $HOME:/mnt/source \
--name="oneformer_v2" sorajang/oneformer_v2 bash

# Git pull orbslam and compile
docker exec -it oneformer bash -i -c "apt-get update && \
    apt-get upgrade && \
    apt-get install libgl1-mesa-dev && \
    apt-get install libglib2.0-0 && \
    conda install pytorch==1.12.0 torchvision==0.13.0 torchaudio==0.12.0 cudatoolkit=11.3 -c pytorch && \
    cd / && \
    git clone --recursive https://github.com/SHI-Labs/OneFormer.git && \
    cd /OneFormer && \
    python tools/setup_detectron2.py && \
    pip3 install git+https://github.com/cocodataset/panopticapi.git && \
    pip3 install git+https://github.com/mcordts/cityscapesScripts.git && \
    pip3 install -r requirements.txt && \
    python -m pip install -e detectron2 && \
    pip3 install mxnet-mkl==1.6.0 numpy==1.23.1"
