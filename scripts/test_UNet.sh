# tube
CUDA_VISIBLE_DEVICES=0 python train.py ./config/UNet/config_tube_prop_bc_geo.yaml -c geo --test | tee ./log/UNet/test/UNet_tube_geo.out
CUDA_VISIBLE_DEVICES=0 python train.py ./config/UNet/config_tube_prop_bc_geo.yaml -c bc --test | tee ./log/UNet/test/UNet_tube_bc.out
CUDA_VISIBLE_DEVICES=0 python train.py ./config/UNet/config_tube_prop_bc_geo.yaml -c prop --test | tee ./log/UNet/test/UNet_tube_prop.out
CUDA_VISIBLE_DEVICES=0 python train.py ./config/UNet/config_tube_prop_bc_geo.yaml -c bc_geo --test | tee ./log/UNet/test/UNet_tube_bc_geo.out
CUDA_VISIBLE_DEVICES=0 python train.py ./config/UNet/config_tube_prop_bc_geo.yaml -c prop_geo --test | tee ./log/UNet/test/UNet_tube_prop_geo.out
CUDA_VISIBLE_DEVICES=0 python train.py ./config/UNet/config_tube_prop_bc_geo.yaml -c prop_bc --test | tee ./log/UNet/test/UNet_tube_prop_bc.out
CUDA_VISIBLE_DEVICES=0 python train.py ./config/UNet/config_tube_prop_bc_geo.yaml -c prop_bc_geo --test | tee ./log/UNet/test/UNet_tube_prop_bc_geo.out

# TGV
CUDA_VISIBLE_DEVICES=0 python train.py ./config/UNet/config_TGV_all.yaml -c Re_ReD --test | tee ./log/UNet/test/UNet_tgv_Re_ReD.out
CUDA_VISIBLE_DEVICES=0 python train.py ./config/UNet/config_TGV_all.yaml -c Re --test | tee ./log/UNet/test/UNet_tgv_Re.out
CUDA_VISIBLE_DEVICES=0 python train.py ./config/UNet/config_TGV_all.yaml -c ReD --test | tee ./log/UNet/test/UNet_tgv_ReD.out

# cavity
CUDA_VISIBLE_DEVICES=0 python train.py ./config/UNet/config_cavity_ReD_bc_re.yaml -c ReD --test | tee ./log/UNet/test/UNet_cavity_ReD.out
CUDA_VISIBLE_DEVICES=0 python train.py ./config/UNet/config_cavity_ReD_bc_re.yaml -c bc --test | tee ./log/UNet/test/UNet_cavity_bc.out
CUDA_VISIBLE_DEVICES=0 python train.py ./config/UNet/config_cavity_ReD_bc_re.yaml -c re --test | tee ./log/UNet/test/UNet_cavity_re.out
CUDA_VISIBLE_DEVICES=0 python train.py ./config/UNet/config_cavity_ReD_bc_re.yaml -c ReD_bc_re --test | tee ./log/UNet/test/UNet_cavity_ReD_bc_re.out

# NSCH
CUDA_VISIBLE_DEVICES=0 python train.py ./config/UNet/config_NSCH_ALL.yaml -c ibc --test | tee ./log/UNet/test/UNet_nsch_ibc.out
CUDA_VISIBLE_DEVICES=0 python train.py ./config/UNet/config_NSCH_ALL.yaml -c ca --test | tee ./log/UNet/test/UNet_nsch_ca.out
CUDA_VISIBLE_DEVICES=0 python train.py ./config/UNet/config_NSCH_ALL.yaml -c mob --test | tee ./log/UNet/test/UNet_nsch_mob.out
CUDA_VISIBLE_DEVICES=0 python train.py ./config/UNet/config_NSCH_ALL.yaml -c phi --test | tee ./log/UNet/test/UNet_nsch_phi.out
CUDA_VISIBLE_DEVICES=0 python train.py ./config/UNet/config_NSCH_ALL.yaml -c re --test | tee ./log/UNet/test/UNet_nsch_re.out
CUDA_VISIBLE_DEVICES=0 python train.py ./config/UNet/config_NSCH_ALL.yaml -c eps --test | tee ./log/UNet/test/UNet_nsch_eps.out
CUDA_VISIBLE_DEVICES=0 python train.py ./config/UNet/config_NSCH_ALL.yaml -c ibc_phi_ca_mob_re_eps --test | tee ./log/UNet/test/UNet_ibc_phi_ca_mob_re_eps.out
CUDA_VISIBLE_DEVICES=0 python train.py ./config/UNet/config_NSCH_ALL.yaml -c ca_mob_re_eps --test | tee ./log/UNet/test/UNet_nsch_ca_mob_re_eps.out
CUDA_VISIBLE_DEVICES=0 python train.py ./config/UNet/config_NSCH_ALL.yaml -c phi_ibc --test | tee ./log/UNet/test/UNet_nsch_phi_ibc.out

# Darcy
CUDA_VISIBLE_DEVICES=0 python train.py ./config/UNet/config_Darcy_PDEBench.yaml -c PDEBench --test | tee ./log/UNet/test/UNet_darcy_PDEBench.out
CUDA_VISIBLE_DEVICES=0 python train.py ./config/UNet/config_Darcy_darcy.yaml -c darcy --test | tee ./log/UNet/test/UNet_darcy_darcy.out

# cylinder
CUDA_VISIBLE_DEVICES=3 python train.py ./config/UNet/config_cylinder_rRE_rBC.yaml -c rRE_rBC --test | tee ./log/UNet/test/UNet_cylinder_rRE_rBC.out
CUDA_VISIBLE_DEVICES=3 python train.py ./config/UNet/config_cylinder_rRE_rBC.yaml -c rRE --test | tee ./log/UNet/test/UNet_cylinder_rRE.out
CUDA_VISIBLE_DEVICES=3 python train.py ./config/UNet/config_cylinder_rRE_rBC.yaml -c rBC --test | tee ./log/UNet/test/UNet_cylinder_rBC.out