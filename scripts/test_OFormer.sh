# tube
CUDA_VISIBLE_DEVICES=1 python train.py ./config/OFormer/config_tube_prop_bc_geo.yaml -c geo --test | tee ./log/test/OFormer_tube_geo.out
CUDA_VISIBLE_DEVICES=1 python train.py ./config/OFormer/config_tube_prop_bc_geo.yaml -c bc --test | tee ./log/test/OFormer_tube_bc.out
CUDA_VISIBLE_DEVICES=1 python train.py ./config/OFormer/config_tube_prop_bc_geo.yaml -c prop --test | tee ./log/test/OFormer_tube_prop.out
CUDA_VISIBLE_DEVICES=1 python train.py ./config/OFormer/config_tube_prop_bc_geo.yaml -c bc_geo --test | tee ./log/test/OFormer_tube_bc_geo.out
CUDA_VISIBLE_DEVICES=1 python train.py ./config/OFormer/config_tube_prop_bc_geo.yaml -c prop_geo --test | tee ./log/test/OFormer_tube_prop_geo.out
CUDA_VISIBLE_DEVICES=1 python train.py ./config/OFormer/config_tube_prop_bc_geo.yaml -c prop_bc --test | tee ./log/test/OFormer_tube_prop_bc.out
CUDA_VISIBLE_DEVICES=1 python train.py ./config/OFormer/config_tube_prop_bc_geo.yaml -c prop_bc_geo --test | tee ./log/test/OFormer_tube_prop_bc_geo.out

# TGV
CUDA_VISIBLE_DEVICES=7 python train.py ./config/OFormer/config_TGV_all.yaml -c Re_ReD --test | tee ./log/test/OFormer_tgv_Re_ReD.out
CUDA_VISIBLE_DEVICES=7 python train.py ./config/OFormer/config_TGV_all.yaml -c Re --test | tee ./log/test/OFormer_tgv_Re.out
CUDA_VISIBLE_DEVICES=7 python train.py ./config/OFormer/config_TGV_all.yaml -c ReD --test | tee ./log/test/OFormer_tgv_ReD.out

# cavity
CUDA_VISIBLE_DEVICES=1 python train.py ./config/OFormer/config_cavity_ReD_bc_re.yaml -c ReD --test | tee ./log/test/OFormer_cavity_ReD.out
CUDA_VISIBLE_DEVICES=1 python train.py ./config/OFormer/config_cavity_ReD_bc_re.yaml -c bc --test | tee ./log/test/OFormer_cavity_bc.out
CUDA_VISIBLE_DEVICES=1 python train.py ./config/OFormer/config_cavity_ReD_bc_re.yaml -c re --test | tee ./log/test/OFormer_cavity_re.out
CUDA_VISIBLE_DEVICES=1 python train.py ./config/OFormer/config_cavity_ReD_bc_re.yaml -c ReD_bc_re --test | tee ./log/test/OFormer_cavity_ReD_bc_re.out

# # NSCH
CUDA_VISIBLE_DEVICES=1 python train.py ./config/OFormer/config_NSCH_ALL.yaml -c ca --test | tee ./log/test/OFormer_nsch_ca.out
CUDA_VISIBLE_DEVICES=1 python train.py ./config/OFormer/config_NSCH_ALL.yaml -c eps --test | tee ./log/test/OFormer_nsch_eps.out
CUDA_VISIBLE_DEVICES=1 python train.py ./config/OFormer/config_NSCH_ALL.yaml -c ibc --test | tee ./log/test/OFormer_nsch_ibc.out
CUDA_VISIBLE_DEVICES=1 python train.py ./config/OFormer/config_NSCH_ALL.yaml -c mob --test | tee ./log/test/OFormer_nsch_mob.out
CUDA_VISIBLE_DEVICES=1 python train.py ./config/OFormer/config_NSCH_ALL.yaml -c phi --test | tee ./log/test/OFormer_nsch_phi.out
CUDA_VISIBLE_DEVICES=1 python train.py ./config/OFormer/config_NSCH_ALL.yaml -c re --test | tee ./log/test/OFormer_nsch_re.out
CUDA_VISIBLE_DEVICES=1 python train.py ./config/OFormer/config_NSCH_ALL.yaml -c ca_mob_re --test | tee ./log/test/OFormer_nsch_ca_mob_re.out
CUDA_VISIBLE_DEVICES=1 python train.py ./config/OFormer/config_NSCH_ALL.yaml -c ibc_phi --test | tee ./log/test/OFormer_nsch_ibc_phi.out
CUDA_VISIBLE_DEVICES=1 python train.py ./config/OFormer/config_NSCH_ALL.yaml -c ca_eps_ibc_mob_phi_re --test | tee ./log/test/OFormer_ca_eps_ibc_mob_phi_re.out

# Darcy
CUDA_VISIBLE_DEVICES=1 python train.py ./config/OFormer/config_Darcy_PDEBench.yaml -c PDEBench --test | tee ./log/test/OFormer_Darcy_PDEBench.out
CUDA_VISIBLE_DEVICES=1 python train.py ./config/OFormer/config_Darcy_darcy.yaml -c darcy --test | tee ./log/test/OFormer_Darcy_darcy.out

cylinder / ircylinder
CUDA_VISIBLE_DEVICES=1 python train.py ./config/OFormer/config_cylinder_irRE_irBC.yaml -c irRE --test | tee ./log/test/OFormer_cylinder_irRE.out
CUDA_VISIBLE_DEVICES=1 python train.py ./config/OFormer/config_cylinder_irRE_irBC.yaml -c irBC --test | tee ./log/test/OFormer_cylinder_irBC.out
CUDA_VISIBLE_DEVICES=1 python train.py ./config/OFormer/config_cylinder_rRE_rBC.yaml -c rRE --test | tee ./log/test/OFormer_cylinder_rRE.out
CUDA_VISIBLE_DEVICES=1 python train.py ./config/OFormer/config_cylinder_rRE_rBC.yaml -c rRE_rBC --test | tee ./log/test/OFormer_cylinder_rRE_rBC.out
CUDA_VISIBLE_DEVICES=1 python train.py ./config/OFormer/config_cylinder_rRE_rBC.yaml -c rBC --test | tee ./log/test/OFormer_cylinder_rBC.out

# hills
CUDA_VISIBLE_DEVICES=1 python train.py ./config/OFormer/config_hills_rRE.yaml -c rRE --test | tee ./log/test/hills_rRE.out
CUDA_VISIBLE_DEVICES=1 python train.py ./config/OFormer/config_hills_irRE.yaml -c irRE --test | tee ./log/test/hills_irRE.out