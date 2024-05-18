# tube
CUDA_VISIBLE_DEVICES=0 python train.py ./config/KNO/config_tube_prop_bc_geo.yaml -c geo --test | tee ./log/test/KNO_tube_geo.out
CUDA_VISIBLE_DEVICES=0 python train.py ./config/KNO/config_tube_prop_bc_geo.yaml -c bc --test | tee ./log/test/KNO_tube_bc.out
CUDA_VISIBLE_DEVICES=0 python train.py ./config/KNO/config_tube_prop_bc_geo.yaml -c prop --test | tee ./log/test/KNO_tube_prop.out
CUDA_VISIBLE_DEVICES=0 python train.py ./config/KNO/config_tube_prop_bc_geo.yaml -c bc_geo --test | tee ./log/test/KNO_tube_bc_geo.out
CUDA_VISIBLE_DEVICES=0 python train.py ./config/KNO/config_tube_prop_bc_geo.yaml -c prop_geo --test | tee ./log/test/KNO_tube_prop_geo.out
CUDA_VISIBLE_DEVICES=0 python train.py ./config/KNO/config_tube_prop_bc_geo.yaml -c prop_bc --test | tee ./log/test/KNO_tube_prop_bc.out
CUDA_VISIBLE_DEVICES=0 python train.py ./config/KNO/config_tube_prop_bc_geo.yaml -c prop_bc_geo --test | tee ./log/test/KNO_tube_prop_bc_geo.out

# TGV
CUDA_VISIBLE_DEVICES=0 python train.py ./config/KNO/config_TGV_all.yaml -c all --test | tee ./log/test/KNO_tgv_all.out
CUDA_VISIBLE_DEVICES=0 python train.py ./config/KNO/config_TGV_all.yaml -c single --test | tee ./log/test/KNO_single.out


# cavity
CUDA_VISIBLE_DEVICES=0 python train.py ./config/KNO/config_cavity_ReD_bc_re.yaml -c ReD --test | tee ./log/test/KNO_cavity_ReD.out
CUDA_VISIBLE_DEVICES=0 python train.py ./config/KNO/config_cavity_ReD_bc_re.yaml -c bc --test | tee ./log/test/KNO_cavity_bc.out
CUDA_VISIBLE_DEVICES=0 python train.py ./config/KNO/config_cavity_ReD_bc_re.yaml -c re --test | tee ./log/test/KNO_cavity_re.out
CUDA_VISIBLE_DEVICES=0 python train.py ./config/KNO/config_cavity_ReD_bc_re.yaml -c ReD_bc_re --test | tee ./log/test/KNO_cavity_ReD_bc_re.out

# NSCH
CUDA_VISIBLE_DEVICES=0 python train.py ./config/KNO/config_NSCH_ALL.yaml -c bc --test | tee ./log/test/KNO_nsch_bc.out
CUDA_VISIBLE_DEVICES=0 python train.py ./config/KNO/config_NSCH_ALL.yaml -c ca --test | tee ./log/test/KNO_nsch_ca.out
CUDA_VISIBLE_DEVICES=0 python train.py ./config/KNO/config_NSCH_ALL.yaml -c mob --test | tee ./log/test/KNO_nsch_mob.out
CUDA_VISIBLE_DEVICES=0 python train.py ./config/KNO/config_NSCH_ALL.yaml -c phi --test | tee ./log/test/KNO_nsch_phi.out
CUDA_VISIBLE_DEVICES=0 python train.py ./config/KNO/config_NSCH_ALL.yaml -c pre --test | tee ./log/test/KNO_nsch_pre.out
CUDA_VISIBLE_DEVICES=0 python train.py ./config/KNO/config_NSCH_ALL.yaml -c re --test | tee ./log/test/KNO_nsch_re.out
CUDA_VISIBLE_DEVICES=0 python train.py ./config/KNO/config_NSCH_ALL.yaml -c uv0 --test | tee ./log/test/KNO_nsch_uv0.out
CUDA_VISIBLE_DEVICES=0 python train.py ./config/KNO/config_NSCH_ALL.yaml -c bc_ca_mob_phi_pre_re_uv0 --test | tee ./log/test/KNO_nsch_bc_ca_mob_phi_pre_re_uv0.out
CUDA_VISIBLE_DEVICES=0 python train.py ./config/KNO/config_NSCH_ALL.yaml -c ca_mob_re --test | tee ./log/test/KNO_nsch_ca_mob_re.out
CUDA_VISIBLE_DEVICES=0 python train.py ./config/KNO/config_NSCH_ALL.yaml -c phi_pre_uv0 --test | tee ./log/test/KNO_nsch_phi_pre_uv0.out
