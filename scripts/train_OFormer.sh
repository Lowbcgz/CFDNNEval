# # tube
# CUDA_VISIBLE_DEVICES=7 python train.py ./config/OFormer/config_tube_prop_bc_geo.yaml -c geo | tee ./log/OFormer/train/OFormer_tube_geo.out
# CUDA_VISIBLE_DEVICES=7 python train.py ./config/OFormer/config_tube_prop_bc_geo.yaml -c bc | tee ./log/OFormer/train/OFormer_tube_bc.out
# CUDA_VISIBLE_DEVICES=7 python train.py ./config/OFormer/config_tube_prop_bc_geo.yaml -c prop | tee ./log/OFormer/train/OFormer_tube_prop.out
# CUDA_VISIBLE_DEVICES=7 python train.py ./config/OFormer/config_tube_prop_bc_geo.yaml -c bc_geo | tee ./log/OFormer/train/OFormer_tube_bc_geo.out
# CUDA_VISIBLE_DEVICES=7 python train.py ./config/OFormer/config_tube_prop_bc_geo.yaml -c prop_geo | tee ./log/OFormer/train/OFormer_tube_prop_geo.out
# CUDA_VISIBLE_DEVICES=7 python train.py ./config/OFormer/config_tube_prop_bc_geo.yaml -c prop_bc | tee ./log/OFormer/train/OFormer_tube_prop_bc.out
# CUDA_VISIBLE_DEVICES=7 python train.py ./config/OFormer/config_tube_prop_bc_geo.yaml -c prop_bc_geo | tee ./log/OFormer/train/OFormer_tube_prop_bc_geo.out

# # TGV
# CUDA_VISIBLE_DEVICES=7 python train.py ./config/OFormer/config_TGV_all.yaml -c all | tee ./log/OFormer/train/OFormer_tgv_all.out
# CUDA_VISIBLE_DEVICES=7 python train.py ./config/OFormer/config_TGV_all.yaml -c single | tee ./log/OFormer/train/OFormer_tgv_single.out

# # cavity
# CUDA_VISIBLE_DEVICES=7 python train.py ./config/OFormer/config_cavity_ReD_bc_re.yaml -c ReD | tee ./log/OFormer/train/OFormer_cavity_ReD.out
# CUDA_VISIBLE_DEVICES=7 python train.py ./config/OFormer/config_cavity_ReD_bc_re.yaml -c bc | tee ./log/OFormer/train/OFormer_cavity_bc.out
# CUDA_VISIBLE_DEVICES=7 python train.py ./config/OFormer/config_cavity_ReD_bc_re.yaml -c re | tee ./log/OFormer/train/OFormer_cavity_re.out
# CUDA_VISIBLE_DEVICES=7 python train.py ./config/OFormer/config_cavity_ReD_bc_re.yaml -c ReD_bc_re | tee ./log/OFormer/train/OFormer_cavity_ReD_bc_re.out

# # NSCH
# CUDA_VISIBLE_DEVICES=5 python train.py ./config/OFormer/config_NSCH_ALL.yaml -c ibc | tee ./log/OFormer/train/OFormer_nsch_ibc.out
# CUDA_VISIBLE_DEVICES=5 python train.py ./config/OFormer/config_NSCH_ALL.yaml -c ca | tee ./log/OFormer/train/OFormer_nsch_ca.out
# CUDA_VISIBLE_DEVICES=5 python train.py ./config/OFormer/config_NSCH_ALL.yaml -c mob | tee ./log/OFormer/train/OFormer_nsch_mob.out
# CUDA_VISIBLE_DEVICES=5 python train.py ./config/OFormer/config_NSCH_ALL.yaml -c phi | tee ./log/OFormer/train/OFormer_nsch_phi.out
# CUDA_VISIBLE_DEVICES=5 python train.py ./config/OFormer/config_NSCH_ALL.yaml -c re | tee ./log/OFormer/train/OFormer_nsch_re.out
# CUDA_VISIBLE_DEVICES=5 python train.py ./config/OFormer/config_NSCH_ALL.yaml -c eps | tee ./log/OFormer/train/OFormer_nsch_eps.out
# CUDA_VISIBLE_DEVICES=5 python train.py ./config/OFormer/config_NSCH_ALL.yaml -c ibc_phi_ca_mob_re_eps | tee ./log/OFormer/train/OFormer_ibc_phi_ca_mob_re_eps.out
# CUDA_VISIBLE_DEVICES=5 python train.py ./config/OFormer/config_NSCH_ALL.yaml -c ca_mob_re_eps | tee ./log/OFormer/train/OFormer_nsch_ca_mob_re_eps.out
# CUDA_VISIBLE_DEVICES=5 python train.py ./config/OFormer/config_NSCH_ALL.yaml -c phi_ibc | tee ./log/OFormer/train/OFormer_nsch_phi_ibc.out

# # Darcy
CUDA_VISIBLE_DEVICES=9 python train.py ./config/OFormer/config_Darcy_PDEBench.yaml -c PDEBench | tee ./log/OFormer/train/OFormer_PDEBench.out
CUDA_VISIBLE_DEVICES=9 python train.py ./config/OFormer/config_Darcy_darcy.yaml -c darcy | tee ./log/OFormer/train/OFormer_PDEBench.out

# # cylinder
# CUDA_VISIBLE_DEVICES=5 python train.py ./config/OFormer/config_cylinder_rRE_rBC.yaml -c rRE_rBC | tee ./log/OFormer/train/OFormer_cylinder_rRE_rBC.out
# CUDA_VISIBLE_DEVICES=5 python train.py ./config/OFormer/config_cylinder_rRE_rBC.yaml -c rRE | tee ./log/OFormer/train/OFormer_cylinder_rRE.out
# CUDA_VISIBLE_DEVICES=5 python train.py ./config/OFormer/config_cylinder_rRE_rBC.yaml -c rBC | tee ./log/OFormer/train/OFormer_cylinder_rBC.out
# CUDA_VISIBLE_DEVICES=5 python train.py ./config/OFormer/config_cylinder_irRE_irBC.yaml -c irRE | tee ./log/OFormer/train/OFormer_cylinder_irRE.out
# CUDA_VISIBLE_DEVICES=5 python train.py ./config/OFormer/config_cylinder_irRE_irBC.yaml -c irBC | tee ./log/OFormer/train/OFormer_cylinder_irBC.out