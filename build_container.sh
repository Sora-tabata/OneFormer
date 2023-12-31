docker stop oneformer_v2
docker rm oneformer_v2



docker pull sorajang/oneformer_v2



# Create a new container
docker run -i -t --gpus all -d \
--shm-size=12gb \
-v $HOME:/mnt/source \
--name="oneformer_v2" sorajang/oneformer_v2 bash

# Git pull orbslam and compile
docker exec -it oneformer_v2 bash -i -c "apt-get update && \
    apt-get upgrade && \
    apt-get install libgl1-mesa-dev && \
    apt-get install libglib2.0-0 && \
    conda install pytorch==2.0.0 torchvision==0.15.0 torchaudio==2.0.0 pytorch-cuda=11.7 -c pytorch -c nvidia && \
    cd /OneFormer && \
    python tools/setup_detectron2.py && \
    pip3 install git+https://github.com/cocodataset/panopticapi.git && \
    pip3 install git+https://github.com/mcordts/cityscapesScripts.git && \
    pip3 install -r requirements.txt && \
    python -m pip install -e detectron2 && \
    pip3 install mxnet-mkl==1.6.0 numpy==1.23.1"
