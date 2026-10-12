# Environment image for the geoglyph experiments (code is NOT baked in;
# the Job downloads train_efficientnetb0.py from GitHub at startup).
FROM tensorflow/tensorflow:2.19.0-gpu

RUN apt-get update && apt-get install -y --no-install-recommends git bash \
    && rm -rf /var/lib/apt/lists/*

RUN pip install --no-cache-dir \
    pandas \
    scikit-learn \
    matplotlib \
    jupyterlab

WORKDIR /workspace
