FROM nvidia/cuda:11.7.1-cudnn8-devel-ubuntu20.04
ARG DEBIAN_FRONTEND=noninteractive
# Install apt-getable dependencies
RUN apt-get update
RUN apt-get install -y python3-pip git build-essential wget gcc make cmake
RUN apt clean
#RUN cd / && git clone --recursive https://github.com/princeton-vl/DROID-SLAM
WORKDIR /opt


RUN wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh && \
    sh Miniconda3-latest-Linux-x86_64.sh -b -p /opt/miniconda3 && \
    rm -r Miniconda3-latest-Linux-x86_64.sh

#COPY ./environment.yaml /environment.yaml
# set path
ENV PATH /opt/miniconda3/bin:$PATH

RUN pip install --upgrade pip && \
    conda update -n base -c defaults conda && \
    conda create --name oneformer python=3.10 -y && \
    #conda env create -f /environment.yaml && \
    conda init && \
    echo "conda activate oneformer" >> ~/.bashrc

ENV CONDA_DEFAULT_ENV oneformer && \
    PATH /opt/conda/envs/oneformer/bin:$PATH

RUN apt-get update
#RUN apt-get install libgl1-mesa-dev


RUN cd /
RUN git clone --recursive https://github.com/SHI-Labs/OneFormer.git /OneFormer
RUN cd /OneFormer
#RUN pip install pytorch==1.10.1 torchvision==0.11.2 cudatoolkit=11.3 -c pytorch -c conda-forge
RUN pip3 install -U opencv-python
#RUN python3 tools/setup_detectron2.py
#RUN pip3 install git+https://github.com/cocodataset/panopticapi.git
#RUN pip3 install git+https://github.com/mcordts/cityscapesScripts.git
#RUN pip3 install -r requirements.txt

RUN pip install gdown
#COPY setup.py .
#ENTRYPOINT ["conda", "run", "--no-capture\-output", "-n", "myenv", "python", "setup.py", "install"]
#RUN python setup.py install


CMD ["bash"]