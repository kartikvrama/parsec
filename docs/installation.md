# Installation

This project was built using python 3.10. We recommend using an anaconda environment for this project. Here are the steps for installing all dependencies:
* Install pytorch based on your system configuration by following instructions in [this link](https://pytorch.org/get-started/locally/).
* Install Torch-Geometric by running `./bin/install-torch-geometric.sh` from within your virtual environment/anaconda environment ([source](https://github.com/kaust-rccl/conda-environment-examples/blob/pytorch-geometric-and-friends/bin/create-conda-env.sh)).
* Install pywFM (for CF+) by running the following ([source](https://github.com/jfloff/pywFM)):
    ```
    git clone https://github.com/srendle/libfm /home/libfm
    cd /home/libfm/
    # taking advantage of a bug to allow us to save model #ShameShame
    git reset --hard 91f8504a15120ef6815d6e10cc7dee42eebaab0f
    make all
    export LIBFM_PATH=/home/libfm/bin/
    ```
* Install the remaining dependencies by running `pip install -r requirements.txt`.