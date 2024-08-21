# cavity
CUDA_VISIBLE_DEVICES=6 python train.py ./config/UNet/config_cavity_ReD_bc_re.yaml -c ReD | tee ./log/UNet/train/UNet_cavity_ReD.out
CUDA_VISIBLE_DEVICES=6 python train.py ./config/UNet/config_cavity_ReD_bc_re.yaml -c bc | tee ./log/UNet/train/UNet_cavity_bc.out
CUDA_VISIBLE_DEVICES=6 python train.py ./config/UNet/config_cavity_ReD_bc_re.yaml -c re | tee ./log/UNet/train/UNet_cavity_re.out
CUDA_VISIBLE_DEVICES=6 python train.py ./config/UNet/config_cavity_ReD_bc_re.yaml -c ReD_bc_re | tee ./log/UNet/train/UNet_cavity_ReD_bc_re.out

# tube
CUDA_VISIBLE_DEVICES=6 python train.py ./config/UNet/config_tube_prop_bc_geo.yaml -c geo | tee ./log/UNet/train/UNet_tube_geo.out
CUDA_VISIBLE_DEVICES=6 python train.py ./config/UNet/config_tube_prop_bc_geo.yaml -c bc | tee ./log/UNet/train/UNet_tube_bc.out
CUDA_VISIBLE_DEVICES=6 python train.py ./config/UNet/config_tube_prop_bc_geo.yaml -c prop | tee ./log/UNet/train/UNet_tube_prop.out
CUDA_VISIBLE_DEVICES=6 python train.py ./config/UNet/config_tube_prop_bc_geo.yaml -c bc_geo | tee ./log/UNet/train/UNet_tube_bc_geo.out
CUDA_VISIBLE_DEVICES=6 python train.py ./config/UNet/config_tube_prop_bc_geo.yaml -c prop_geo | tee ./log/UNet/train/UNet_tube_prop_geo.out
CUDA_VISIBLE_DEVICES=6 python train.py ./config/UNet/config_tube_prop_bc_geo.yaml -c prop_bc | tee ./log/UNet/train/UNet_tube_prop_bc.out
CUDA_VISIBLE_DEVICES=6 python train.py ./config/UNet/config_tube_prop_bc_geo.yaml -c prop_bc_geo | tee ./log/UNet/train/UNet_tube_prop_bc_geo.out

# TGV
CUDA_VISIBLE_DEVICES=0 python train.py ./config/UNet/config_TGV_all.yaml -c Re_ReD | tee ./log/UNet/train/UNet_tgv_Re_ReD.out
CUDA_VISIBLE_DEVICES=0 python train.py ./config/UNet/config_TGV_all.yaml -c Re | tee ./log/UNet/train/UNet_tgv_Re.out
CUDA_VISIBLE_DEVICES=0 python train.py ./config/UNet/config_TGV_all.yaml -c ReD | tee ./log/UNet/train/UNet_tgv_ReD.out

# NSCH
CUDA_VISIBLE_DEVICES=0 python train.py ./config/UNet/config_NSCH_ALL.yaml -c ibc | tee ./log/UNet/train/UNet_nsch_ibc.out
CUDA_VISIBLE_DEVICES=0 python train.py ./config/UNet/config_NSCH_ALL.yaml -c ca | tee ./log/UNet/train/UNet_nsch_ca.out
CUDA_VISIBLE_DEVICES=0 python train.py ./config/UNet/config_NSCH_ALL.yaml -c mob | tee ./log/UNet/train/UNet_nsch_mob.out
CUDA_VISIBLE_DEVICES=0 python train.py ./config/UNet/config_NSCH_ALL.yaml -c phi | tee ./log/UNet/train/UNet_nsch_phi.out
CUDA_VISIBLE_DEVICES=0 python train.py ./config/UNet/config_NSCH_ALL.yaml -c re | tee ./log/UNet/train/UNet_nsch_re.out
CUDA_VISIBLE_DEVICES=0 python train.py ./config/UNet/config_NSCH_ALL.yaml -c eps | tee ./log/UNet/train/UNet_nsch_eps.out
CUDA_VISIBLE_DEVICES=0 python train.py ./config/UNet/config_NSCH_ALL.yaml -c ibc_phi_ca_mob_re_eps | tee ./log/UNet/train/UNet_ibc_phi_ca_mob_re_eps.out
CUDA_VISIBLE_DEVICES=0 python train.py ./config/UNet/config_NSCH_ALL.yaml -c ca_mob_re_eps | tee ./log/UNet/train/UNet_nsch_ca_mob_re_eps.out
CUDA_VISIBLE_DEVICES=0 python train.py ./config/UNet/config_NSCH_ALL.yaml -c phi_ibc | tee ./log/UNet/train/UNet_nsch_phi_ibc.out

# Darcy
CUDA_VISIBLE_DEVICES=0 python train.py ./config/UNet/config_Darcy_PDEBench.yaml -c PDEBench | tee ./log/UNet/train/UNet_darcy_PDEBench.out
CUDA_VISIBLE_DEVICES=0 python train.py ./config/UNet/config_Darcy_darcy.yaml -c darcy | tee ./log/UNet/train/UNet_darcy_darcy.out

# cylinder
CUDA_VISIBLE_DEVICES=3 python train.py ./config/UNet/config_cylinder_rRE_rBC.yaml -c rRE_rBC | tee ./log/UNet/train/UNet_cylinder_rRE_rBC.out
CUDA_VISIBLE_DEVICES=3 python train.py ./config/UNet/config_cylinder_rRE_rBC.yaml -c rRE | tee ./log/UNet/train/UNet_cylinder_rRE.out
CUDA_VISIBLE_DEVICES=3 python train.py ./config/UNet/config_cylinder_rRE_rBC.yaml -c rBC | tee ./log/UNet/train/UNet_cylinder_rBC.out

# periodic hills
CUDA_VISIBLE_DEVICES=3 python train.py ./config/UNet/config_hills_rRE.yaml -c rRE | tee ./log/UNet/train/UNet_hills_rRE.out