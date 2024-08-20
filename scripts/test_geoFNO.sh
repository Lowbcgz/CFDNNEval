# cylinder / ircylinder
CUDA_VISIBLE_DEVICES=0 python train.py ./config/geoFNO/config_ircylinder_irRE_irBC.yaml -c irRE --test | tee ./log/test/geoFNO_ircylinder_irRE.out
CUDA_VISIBLE_DEVICES=0 python train.py ./config/geoFNO/config_ircylinder_irRE_irBC.yaml -c irBC --test | tee ./log/test/geoFNO_ircylinder_irBC.out
