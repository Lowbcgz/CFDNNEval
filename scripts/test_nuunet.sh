
# cylinder / ircylinder
CUDA_VISIBLE_DEVICES=0 python train.py ./config/NUUNet/config_ircylinder_irRE_irBC.yaml -c irRE --test | tee ./log/test/NUUNet_ircylinder_irRE.out
CUDA_VISIBLE_DEVICES=0 python train.py ./config/NUUNet/config_ircylinder_irRE_irBC.yaml -c irBC --test | tee ./log/test/NUUNet_ircylinder_irBC.out

# periodic hills
CUDA_VISIBLE_DEVICES=6 python train.py ./config/NUUNet/config_irhills_irRE.yaml -c irRE --test | tee ./log/test/NUUNet_irhills_irRE.out
