# cylinder / ircylinder
CUDA_VISIBLE_DEVICES=0 python train.py ./config/NUFNO/config_ircylinder_irRE_irBC.yaml -c irRE | tee ./log/train/NUFNO_ircylinder_irRE.out
CUDA_VISIBLE_DEVICES=0 python train.py ./config/NUFNO/config_ircylinder_irRE_irBC.yaml -c irBC | tee ./log/train/NUFNO_ircylinder_irBC.out

# periodic hills
CUDA_VISIBLE_DEVICES=0 python train.py ./config/NUFNO/config_irhills_irRE.yaml -c irRE | tee ./log/train/NUFNO_irhills_irRE.out