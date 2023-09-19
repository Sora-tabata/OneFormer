docker stop oneformer
docker rm oneformer



docker pull sorajang/oneformer



# Create a new container
docker run -i -t --gpus all -d \
--shm-size=12gb \
-v $HOME:/mnt/source \
--name="oneformer" sorajang/oneformer bash

# Git pull orbslam and compile
docker exec -it oneformer bash -i -c "apt-get update && \
    apt-get upgrade && \
    apt-get install libgl1-mesa-dev && \
    apt-get install libglib2.0-0 && \
    conda install pytorch==1.10.1 torchvision==0.11.2 cudatoolkit=11.3 -c pytorch -c conda-forge && \
    cd /OneFormer && \
    python tools/setup_detectron2.py && \
    pip3 install git+https://github.com/cocodataset/panopticapi.git && \
    pip3 install git+https://github.com/mcordts/cityscapesScripts.git && \
    pip3 install -r requirements.txt && \
    python -m pip install 'git+https://github.com/facebookresearch/detectron2.git' && \
    pip3 install mxnet-mkl==1.6.0 numpy==1.23.1"
