# # tube
CUDA_VISIBLE_DEVICES=0 python train.py ./config/DeepONet/config_tube_prop_bc_geo.yaml -c geo | tee ./log/train/DeepONet_tube_geo.out
CUDA_VISIBLE_DEVICES=0 python train.py ./config/DeepONet/config_tube_prop_bc_geo.yaml -c bc | tee ./log/train/DeepONet_tube_bc.out
CUDA_VISIBLE_DEVICES=0 python train.py ./config/DeepONet/config_tube_prop_bc_geo.yaml -c prop | tee ./log/train/DeepONet_tube_prop.out
CUDA_VISIBLE_DEVICES=0 python train.py ./config/DeepONet/config_tube_prop_bc_geo.yaml -c bc_geo | tee ./log/train/DeepONet_tube_bc_geo.out
CUDA_VISIBLE_DEVICES=0 python train.py ./config/DeepONet/config_tube_prop_bc_geo.yaml -c prop_geo | tee ./log/train/DeepONet_tube_prop_geo.out
CUDA_VISIBLE_DEVICES=0 python train.py ./config/DeepONet/config_tube_prop_bc_geo.yaml -c prop_bc | tee ./log/train/DeepONet_tube_prop_bc.out
CUDA_VISIBLE_DEVICES=0 python train.py ./config/DeepONet/config_tube_prop_bc_geo.yaml -c prop_bc_geo | tee ./log/train/DeepONet_tube_prop_bc_geo.out

# # TGV
CUDA_VISIBLE_DEVICES=0 python train.py ./config/DeepONet/config_TGV_all.yaml -c all | tee ./log/train/DeepONet_tgv_all.out
CUDA_VISIBLE_DEVICES=0 python train.py ./config/DeepONet/config_TGV_all.yaml -c single | tee ./log/train/DeepONet_tgv_single.out

# # cavity
CUDA_VISIBLE_DEVICES=0 python train.py ./config/DeepONet/config_cavity_ReD_bc_re.yaml -c ReD | tee ./log/train/DeepONet_cavity_ReD.out
CUDA_VISIBLE_DEVICES=0 python train.py ./config/DeepONet/config_cavity_ReD_bc_re.yaml -c bc | tee ./log/train/DeepONet_cavity_bc.out
CUDA_VISIBLE_DEVICES=0 python train.py ./config/DeepONet/config_cavity_ReD_bc_re.yaml -c re | tee ./log/train/DeepONet_cavity_re.out
CUDA_VISIBLE_DEVICES=0 python train.py ./config/DeepONet/config_cavity_ReD_bc_re.yaml -c ReD_bc_re | tee ./log/train/DeepONet_cavity_ReD_bc_re.out

# # NSCH
CUDA_VISIBLE_DEVICES=0 python train.py ./config/DeepONet/config_NSCH_ALL.yaml -c bc | tee ./log/train/DeepONet_nsch_bc.out
CUDA_VISIBLE_DEVICES=0 python train.py ./config/DeepONet/config_NSCH_ALL.yaml -c ca | tee ./log/train/DeepONet_nsch_ca.out
CUDA_VISIBLE_DEVICES=0 python train.py ./config/DeepONet/config_NSCH_ALL.yaml -c mob | tee ./log/train/DeepONet_nsch_mob.out
CUDA_VISIBLE_DEVICES=0 python train.py ./config/DeepONet/config_NSCH_ALL.yaml -c phi | tee ./log/train/DeepONet_nsch_phi.out
CUDA_VISIBLE_DEVICES=0 python train.py ./config/DeepONet/config_NSCH_ALL.yaml -c pre | tee ./log/train/DeepONet_nsch_pre.out
CUDA_VISIBLE_DEVICES=0 python train.py ./config/DeepONet/config_NSCH_ALL.yaml -c re | tee ./log/train/DeepONet_nsch_re.out
CUDA_VISIBLE_DEVICES=0 python train.py ./config/DeepONet/config_NSCH_ALL.yaml -c uv0 | tee ./log/train/DeepONet_nsch_uv0.out
CUDA_VISIBLE_DEVICES=0 python train.py ./config/DeepONet/config_NSCH_ALL.yaml -c bc_ca_mob_phi_pre_re_uv0 | tee ./log/train/DeepONet_nsch_bc_ca_mob_phi_pre_re_uv0.out
CUDA_VISIBLE_DEVICES=0 python train.py ./config/DeepONet/config_NSCH_ALL.yaml -c ca_mob_re | tee ./log/train/DeepONet_nsch_ca_mob_re.out
CUDA_VISIBLE_DEVICES=0 python train.py ./config/DeepONet/config_NSCH_ALL.yaml -c phi_pre_uv0 | tee ./log/train/DeepONet_nsch_phi_pre_uv0.out

# # Darcy
# CUDA_VISIBLE_DEVICES=0 python train.py ./config/DeepONet/config_Darcy_ar_as.yaml -c ar | tee ./log/train/DeepONet_darcy_ar.out
# CUDA_VISIBLE_DEVICES=0 python train.py ./config/DeepONet/config_Darcy_ar_as.yaml -c as | tee ./log/train/DeepONet_darcy_as.out
CUDA_VISIBLE_DEVICES=7 python train.py ./config/DeepONet/config_Darcy_PDEBench.yaml -c PDEBench | tee ./log/train/DeepONet_darcy_PDEBench.out

# cylinder / ircylinder
CUDA_VISIBLE_DEVICES=7 python train.py ./config/DeepONet/config_ircylinder_irRE_irBC.yaml -c irRE | tee ./log/train/DeepONet_ircylinder_irRE.out
CUDA_VISIBLE_DEVICES=7 python train.py ./config/DeepONet/config_ircylinder_irRE_irBC.yaml -c irBC | tee ./log/train/DeepONet_ircylinder_irBC.out
CUDA_VISIBLE_DEVICES=3 python train.py ./config/DeepONet/config_cylinder_rRE_rBC.yaml -c rRE | tee ./log/train/DeepONet_cylinder_rRE.out
CUDA_VISIBLE_DEVICES=6 python train.py ./config/DeepONet/config_cylinder_rRE_rBC.yaml -c rRE_rBC | tee ./log/train/DeepONet_cylinder_rRE_rBC.out
CUDA_VISIBLE_DEVICES=6 python train.py ./config/DeepONet/config_cylinder_rRE_rBC.yaml -c rBC | tee ./log/train/DeepONet_cylinder_rBC.out