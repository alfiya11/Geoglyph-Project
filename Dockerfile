# Environment image for the geoglyph experiments (code is NOT baked in;
# the Job downloads train_efficientnetb0.py from GitHub at startup).
FROM tensorflow/tensorflow:2.19.0-gpu
 
RUN apt-get update && apt-get install -y --no-install-recommends git bash \
    && rm -rf /var/lib/apt/lists/*
 
# The base image ships cuDNN 8, but TensorFlow 2.19 needs cuDNN 9.
# [and-cuda] installs the matching CUDA/cuDNN libraries from pip.
RUN pip install --no-cache-dir "tensorflow[and-cuda]==2.19.0"
 
RUN pip install --no-cache-dir \
    pandas \
    scikit-learn \
    matplotlib \
    jupyterlab
 
# Fail the build if the cuDNN 9 package is missing.
RUN pip list 2>/dev/null | grep -i "nvidia-cudnn-cu12"
 
WORKDIR /workspace
