# tube
CUDA_VISIBLE_DEVICES=0 python train.py ./config/LSM/config_tube_prop_bc_geo.yaml -c geo | tee ./log/train/LSM_tube_geo.out
CUDA_VISIBLE_DEVICES=0 python train.py ./config/LSM/config_tube_prop_bc_geo.yaml -c bc | tee ./log/train/LSM_tube_bc.out
CUDA_VISIBLE_DEVICES=0 python train.py ./config/LSM/config_tube_prop_bc_geo.yaml -c prop | tee ./log/train/LSM_tube_prop.out
CUDA_VISIBLE_DEVICES=0 python train.py ./config/LSM/config_tube_prop_bc_geo.yaml -c bc_geo | tee ./log/train/LSM_tube_bc_geo.out
CUDA_VISIBLE_DEVICES=0 python train.py ./config/LSM/config_tube_prop_bc_geo.yaml -c prop_geo | tee ./log/train/LSM_tube_prop_geo.out
CUDA_VISIBLE_DEVICES=0 python train.py ./config/LSM/config_tube_prop_bc_geo.yaml -c prop_bc | tee ./log/train/LSM_tube_prop_bc.out
CUDA_VISIBLE_DEVICES=0 python train.py ./config/LSM/config_tube_prop_bc_geo.yaml -c prop_bc_geo | tee ./log/train/LSM_tube_prop_bc_geo.out
 
# TGV
CUDA_VISIBLE_DEVICES=0 python train.py ./config/LSM/config_TGV_all.yaml -c all | tee ./log/train/LSM_tgv_all.out
CUDA_VISIBLE_DEVICES=0 python train.py ./config/LSM/config_TGV_all.yaml -c single | tee ./log/train/LSM_tgv_single.out

# cavity
CUDA_VISIBLE_DEVICES=0 python train.py ./config/LSM/config_cavity_ReD_bc_re.yaml -c ReD | tee ./log/train/LSM_cavity_ReD.out
CUDA_VISIBLE_DEVICES=0 python train.py ./config/LSM/config_cavity_ReD_bc_re.yaml -c bc | tee ./log/train/LSM_cavity_bc.out
CUDA_VISIBLE_DEVICES=0 python train.py ./config/LSM/config_cavity_ReD_bc_re.yaml -c re | tee ./log/train/LSM_cavity_re.out
CUDA_VISIBLE_DEVICES=0 python train.py ./config/LSM/config_cavity_ReD_bc_re.yaml -c ReD_bc_re | tee ./log/train/LSM_cavity_ReD_bc_re.out

# # NSCH
CUDA_VISIBLE_DEVICES=0 python train.py ./config/LSM/config_NSCH_ALL.yaml -c ibc | tee ./log/train/LSM_nsch_ibc.out
CUDA_VISIBLE_DEVICES=0 python train.py ./config/LSM/config_NSCH_ALL.yaml -c ca | tee ./log/train/LSM_nsch_ca.out
CUDA_VISIBLE_DEVICES=0 python train.py ./config/LSM/config_NSCH_ALL.yaml -c mob | tee ./log/train/LSM_nsch_mob.out
CUDA_VISIBLE_DEVICES=0 python train.py ./config/LSM/config_NSCH_ALL.yaml -c phi | tee ./log/train/LSM_nsch_phi.out
CUDA_VISIBLE_DEVICES=0 python train.py ./config/LSM/config_NSCH_ALL.yaml -c re | tee ./log/train/LSM_nsch_re.out
CUDA_VISIBLE_DEVICES=0 python train.py ./config/LSM/config_NSCH_ALL.yaml -c eps | tee ./log/train/LSM_nsch_eps.out
CUDA_VISIBLE_DEVICES=0 python train.py ./config/LSM/config_NSCH_ALL.yaml -c ibc_phi_ca_mob_re_eps | tee ./log/train/LSM_ibc_phi_ca_mob_re_eps.out
CUDA_VISIBLE_DEVICES=0 python train.py ./config/LSM/config_NSCH_ALL.yaml -c ca_mob_re_eps | tee ./log/train/LSM_nsch_ca_mob_re_eps.out
CUDA_VISIBLE_DEVICES=0 python train.py ./config/LSM/config_NSCH_ALL.yaml -c phi_ibc | tee ./log/train/LSM_nsch_phi_ibc.out

# Darcy
CUDA_VISIBLE_DEVICES=0 python train.py ./config/LSM/config_Darcy_PDEBench.yaml -c PDEBench | tee ./log/train/LSM_darcy_PDEBench.out
CUDA_VISIBLE_DEVICES=0 python train.py ./config/LSM/config_Darcy_darcy.yaml -c darcy | tee ./log/train/LSM_darcy_darcy.out

# cylinder / ircylinder
CUDA_VISIBLE_DEVICES=0 python train.py ./config/LSM/config_ircylinder_irRE_irBC.yaml -c irRE | tee ./log/train/LSM_ircylinder_irRE.out
CUDA_VISIBLE_DEVICES=0 python train.py ./config/LSM/config_ircylinder_irRE_irBC.yaml -c irBC | tee ./log/train/LSM_ircylinder_irBC.out
CUDA_VISIBLE_DEVICES=0 python train.py ./config/LSM/config_cylinder_rRE_rBC.yaml -c rRE | tee ./log/train/LSM_cylinder_rRE.out
CUDA_VISIBLE_DEVICES=0 python train.py ./config/LSM/config_cylinder_rRE_rBC.yaml -c rRE_rBC | tee ./log/train/LSM_cylinder_rRE_rBC.out
CUDA_VISIBLE_DEVICES=0 python train.py ./config/LSM/config_cylinder_rRE_rBC.yaml -c rBC | tee ./log/train/LSM_cylinder_rBC.out

# periodic hills
CUDA_VISIBLE_DEVICES=0 python train.py ./config/LSM/config_hills_rRE.yaml -c rRE | tee ./log/train/LSM_hills_rRE.out