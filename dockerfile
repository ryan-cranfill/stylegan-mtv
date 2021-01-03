# Copyright (c) 2020, NVIDIA CORPORATION.  All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.

# FROM tensorflow/tensorflow:1.14.0-gpu-py3
FROM nvcr.io/nvidia/tensorflow:20.12-tf1-py3
# FROM tensorflow/tensorflow:1.15.4-gpu-py3

RUN apt-get update -qq
RUN apt-get install -y libgl1-mesa-glx ffmpeg -qq

RUN pip install --upgrade pip
RUN pip install opencv-python==4.4.0.46
RUN pip install scipy==1.3.3
RUN pip install requests==2.22.0
RUN pip install Pillow==6.2.1
RUN pip install h5py==2.9.0
RUN pip install imageio==2.9.0
RUN pip install imageio-ffmpeg==0.4.2
RUN pip install tqdm==4.49.0
RUN pip install youtube-dl==2020.12.22
RUN pip install click==7.1.2
RUN pip install pafy==0.5.5
RUN pip install ffmpeg-python==0.2.0
RUN pip install librosa==0.8.0
RUN pip install spacy==2.3.5

WORKDIR /code
RUN mkdir input
RUN mkdir output
# COPY . .
