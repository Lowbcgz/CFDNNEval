# tube
CUDA_VISIBLE_DEVICES=0 python train.py ./config/FNO/config_tube_prop_bc_geo.yaml -c geo --test | tee ./log/train/tube_geo.out
CUDA_VISIBLE_DEVICES=0 python train.py ./config/FNO/config_tube_prop_bc_geo.yaml -c bc --test | tee ./log/train/tube_bc.out
CUDA_VISIBLE_DEVICES=0 python train.py ./config/FNO/config_tube_prop_bc_geo.yaml -c prop --test | tee ./log/train/tube_prop.out
CUDA_VISIBLE_DEVICES=0 python train.py ./config/FNO/config_tube_prop_bc_geo.yaml -c bc_geo --test | tee ./log/train/tube_bc_geo.out
CUDA_VISIBLE_DEVICES=0 python train.py ./config/FNO/config_tube_prop_bc_geo.yaml -c prop_geo --test | tee ./log/train/tube_prop_geo.out
CUDA_VISIBLE_DEVICES=0 python train.py ./config/FNO/config_tube_prop_bc_geo.yaml -c prop_bc --test | tee ./log/train/tube_prop_bc.out
CUDA_VISIBLE_DEVICES=0 python train.py ./config/FNO/config_tube_prop_bc_geo.yaml -c prop_bc_geo --test | tee ./log/train/tube_prop_bc_geo.out