# cylinder / ircylinder
CUDA_VISIBLE_DEVICES=6 python train.py ./config/NUUNet/config_cylinder_irRE_irBC.yaml -c irRE | tee ./log/train/NUUNet_cylinder_irRE.out
CUDA_VISIBLE_DEVICES=6 python train.py ./config/NUUNet/config_cylinder_irRE_irBC.yaml -c irBC | tee ./log/train/NUUNet_cylinder_irBC.out

# periodic hills
CUDA_VISIBLE_DEVICES=0 python train.py ./config/NUUNet/config_irhills_irRE.yaml -c irRE | tee ./log/train/NUUNet_irhills_irRE.out