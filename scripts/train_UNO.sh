# tube
CUDA_VISIBLE_DEVICES=7 python train.py ./config/UNO/config_tube_prop_bc_geo.yaml -c geo | tee ./log/train/UNO_tube_geo.out
CUDA_VISIBLE_DEVICES=7 python train.py ./config/UNO/config_tube_prop_bc_geo.yaml -c bc | tee ./log/train/UNO_tube_bc.out
CUDA_VISIBLE_DEVICES=7 python train.py ./config/UNO/config_tube_prop_bc_geo.yaml -c prop | tee ./log/train/UNO_tube_prop.out
CUDA_VISIBLE_DEVICES=7 python train.py ./config/UNO/config_tube_prop_bc_geo.yaml -c bc_geo | tee ./log/train/UNO_tube_bc_geo.out
CUDA_VISIBLE_DEVICES=7 python train.py ./config/UNO/config_tube_prop_bc_geo.yaml -c prop_geo | tee ./log/train/UNO_tube_prop_geo.out
CUDA_VISIBLE_DEVICES=7 python train.py ./config/UNO/config_tube_prop_bc_geo.yaml -c prop_bc | tee ./log/train/UNO_tube_prop_bc.out
CUDA_VISIBLE_DEVICES=7 python train.py ./config/UNO/config_tube_prop_bc_geo.yaml -c prop_bc_geo | tee ./log/train/UNO_tube_prop_bc_geo.out

# TGV
CUDA_VISIBLE_DEVICES=7 python train.py ./config/UNO/config_TGV_ALL.yaml -c rho_V0_nu | tee ./log/train/UNO_tgv_rho_V0_nu.out
CUDA_VISIBLE_DEVICES=7 python train.py ./config/UNO/config_TGV_ALL.yaml -c rho | tee ./log/train/UNO_tgv_rho.out
CUDA_VISIBLE_DEVICES=7 python train.py ./config/UNO/config_TGV_ALL.yaml -c V0 | tee ./log/train/UNO_tgv_V0.out
CUDA_VISIBLE_DEVICES=7 python train.py ./config/UNO/config_TGV_ALL.yaml -c nu | tee ./log/train/UNO_tgv_nu.out

# cavity
CUDA_VISIBLE_DEVICES=7 python train.py ./config/UNO/config_cavity_ReD_bc_re.yaml -c ReD | tee ./log/train/UNO_cavity_ReD.out
CUDA_VISIBLE_DEVICES=7 python train.py ./config/UNO/config_cavity_ReD_bc_re.yaml -c bc | tee ./log/train/UNO_cavity_bc.out
CUDA_VISIBLE_DEVICES=7 python train.py ./config/UNO/config_cavity_ReD_bc_re.yaml -c re | tee ./log/train/UNO_cavity_re.out
CUDA_VISIBLE_DEVICES=7 python train.py ./config/UNO/config_cavity_ReD_bc_re.yaml -c ReD_bc_re | tee ./log/train/UNO_cavity_ReD_bc_re.out

# NSCH
CUDA_VISIBLE_DEVICES=7 python train.py ./config/UNO/config_NSCH_ALL.yaml -c bc | tee ./log/train/UNO_nsch_bc.out
CUDA_VISIBLE_DEVICES=7 python train.py ./config/UNO/config_NSCH_ALL.yaml -c ca | tee ./log/train/UNO_nsch_ca.out
CUDA_VISIBLE_DEVICES=7 python train.py ./config/UNO/config_NSCH_ALL.yaml -c mob | tee ./log/train/UNO_nsch_mob.out
CUDA_VISIBLE_DEVICES=7 python train.py ./config/UNO/config_NSCH_ALL.yaml -c phi | tee ./log/train/UNO_nsch_phi.out
CUDA_VISIBLE_DEVICES=7 python train.py ./config/UNO/config_NSCH_ALL.yaml -c pre | tee ./log/train/UNO_nsch_pre.out
CUDA_VISIBLE_DEVICES=7 python train.py ./config/UNO/config_NSCH_ALL.yaml -c re | tee ./log/train/UNO_nsch_re.out
CUDA_VISIBLE_DEVICES=7 python train.py ./config/UNO/config_NSCH_ALL.yaml -c uv0 | tee ./log/train/UNO_nsch_uv0.out
CUDA_VISIBLE_DEVICES=7 python train.py ./config/UNO/config_NSCH_ALL.yaml -c bc_ca_mob_phi_pre_re_uv0 | tee ./log/train/UNO_nsch_bc_ca_mob_phi_pre_re_uv0.out
CUDA_VISIBLE_DEVICES=7 python train.py ./config/UNO/config_NSCH_ALL.yaml -c ca_mob_re | tee ./log/train/UNO_nsch_ca_mob_re.out
CUDA_VISIBLE_DEVICES=7 python train.py ./config/UNO/config_NSCH_ALL.yaml -c phi_pre_uv0 | tee ./log/train/UNO_nsch_phi_pre_uv0.out

# Darcy
CUDA_VISIBLE_DEVICES=7 python train.py ./config/UNO/config_Darcy_ar_as.yaml -c ar | tee ./log/train/UNO_darcy_ar.out
CUDA_VISIBLE_DEVICES=7 python train.py ./config/UNO/config_Darcy_ar_as.yaml -c as | tee ./log/train/UNO_darcy_as.out
CUDA_VISIBLE_DEVICES=7 python train.py ./config/UNO/config_Darcy_ar_as.yaml -c as_ar | tee ./log/train/UNO_darcy_as_ar.out