# cylinder / ircylinder
CUDA_VISIBLE_DEVICES=0 python train.py ./config/NUFNO/config_ircylinder_irRE_irBC.yaml -c irRE --test | tee ./log/test/NUFNO_ircylinder_irRE.out
CUDA_VISIBLE_DEVICES=0 python train.py ./config/NUFNO/config_ircylinder_irRE_irBC.yaml -c irBC --test | tee ./log/test/NUFNO_ircylinder_irBC.out

# periodic hills
CUDA_VISIBLE_DEVICES=6 python train.py ./config/NUFNO/config_irhills_irRE.yaml -c irRE --test | tee ./log/test/NUFNO_irhills_irRE.out
