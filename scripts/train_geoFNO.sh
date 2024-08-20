# cylinder / ircylinder
CUDA_VISIBLE_DEVICES=0 python train.py ./config/geoFNO/config_ircylinder_irRE_irBC.yaml -c irRE | tee ./log/train/geoFNO_ircylinder_irRE.out
CUDA_VISIBLE_DEVICES=0 python train.py ./config/geoFNO/config_ircylinder_irRE_irBC.yaml -c irBC | tee ./log/train/geoFNO_ircylinder_irBC.out

