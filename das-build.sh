#! /bin/bash
git submodule update --init
pip3 install -r requirements.txt --trusted-host 10.68.20.101

export LD_LIBRARY_PATH=/opt/rocm/lib:$LD_LIBRARY_PATH
PREBUILD_KERNELS=1 GPU_ARCHS="gfx936;gfx938" python3 setup.py bdist_wheel
