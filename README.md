# Stereo Match Network
Object's disparity with centernet as the backbone.


# Installation

environment is same as center net, please refer to [CenterNet](https://github.com/xingyizhou/CenterNet/blob/master/readme/INSTALL.md)

Install pytorch 1.0.1 with cuda 10

Cuda 10 with pytorch 1.0:

    conda install pytorch=1.0.1 torchvision -c pytorch



Compile deformable convolutional with cuda 10

    If you use Cuda 10 try to

    ```
    1. git clone git@github.com:CharlesShang/DCNv2.git
    2. before execute make./sh, check pytorch version.
    Probably not 1.0, serveral things need to check ~/.local/lib/python3.6/site_packages/torch
    python -c "import sys; print(sys.path)" check the system path.
    3. do ./make.sh
    ```
