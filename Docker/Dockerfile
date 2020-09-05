# TODO - use the image with PyTorch and some tools installed
FROM ufoym/deepo:pytorch-py36-cu101
RUN apt-get update

# TODO - prepare locally installing DMLab
ADD /other_module /other_module

# TODO - installing DMLab from local pkg + other packages
RUN pip install /other_module/deepmind_lab-1.0-py3-none-any.whl --force-reinstall \
    && apt-get -y install libffi-dev gettext freeglut3-dev libsdl2-dev \
      zip libosmesa6-dev python-dev python-numpy python-pil python3-dev \
      python3-numpy python3-pil \
    && pip install dm_env \
    && pip install ray gym hydra-core
#    && rm -r /other_module