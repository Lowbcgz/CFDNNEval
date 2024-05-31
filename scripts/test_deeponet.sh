# # tube
CUDA_VISIBLE_DEVICES=0 python train.py ./config/DeepONet/config_tube_prop_bc_geo.yaml -c geo --test | tee ./log/test/DeepONet_tube_geo.out
CUDA_VISIBLE_DEVICES=0 python train.py ./config/DeepONet/config_tube_prop_bc_geo.yaml -c bc --test | tee ./log/test/DeepONet_tube_bc.out
CUDA_VISIBLE_DEVICES=0 python train.py ./config/DeepONet/config_tube_prop_bc_geo.yaml -c prop --test | tee ./log/test/DeepONet_tube_prop.out
CUDA_VISIBLE_DEVICES=0 python train.py ./config/DeepONet/config_tube_prop_bc_geo.yaml -c bc_geo --test | tee ./log/test/DeepONet_tube_bc_geo.out
CUDA_VISIBLE_DEVICES=0 python train.py ./config/DeepONet/config_tube_prop_bc_geo.yaml -c prop_geo --test | tee ./log/test/DeepONet_tube_prop_geo.out
CUDA_VISIBLE_DEVICES=0 python train.py ./config/DeepONet/config_tube_prop_bc_geo.yaml -c prop_bc --test | tee ./log/test/DeepONet_tube_prop_bc.out
CUDA_VISIBLE_DEVICES=0 python train.py ./config/DeepONet/config_tube_prop_bc_geo.yaml -c prop_bc_geo --test | tee ./log/test/DeepONet_tube_prop_bc_geo.out

# # TGV
CUDA_VISIBLE_DEVICES=0 python train.py ./config/DeepONet/config_TGV_all.yaml -c all --test | tee ./log/test/DeepONet_tgv_all.out
CUDA_VISIBLE_DEVICES=0 python train.py ./config/DeepONet/config_TGV_all.yaml -c single --test | tee ./log/test/DeepONet_tgv_single.out

# # cavity
CUDA_VISIBLE_DEVICES=0 python train.py ./config/DeepONet/config_cavity_ReD_bc_re.yaml -c ReD --test | tee ./log/test/DeepONet_cavity_ReD.out
CUDA_VISIBLE_DEVICES=0 python train.py ./config/DeepONet/config_cavity_ReD_bc_re.yaml -c bc --test | tee ./log/test/DeepONet_cavity_bc.out
CUDA_VISIBLE_DEVICES=0 python train.py ./config/DeepONet/config_cavity_ReD_bc_re.yaml -c re --test | tee ./log/test/DeepONet_cavity_re.out
CUDA_VISIBLE_DEVICES=0 python train.py ./config/DeepONet/config_cavity_ReD_bc_re.yaml -c ReD_bc_re --test | tee ./log/test/DeepONet_cavity_ReD_bc_re.out

# # NSCH
CUDA_VISIBLE_DEVICES=0 python train.py ./config/DeepONet/config_NSCH_ALL.yaml -c ibc --test | tee ./log/test/DeepONet_nsch_ibc.out
CUDA_VISIBLE_DEVICES=0 python train.py ./config/DeepONet/config_NSCH_ALL.yaml -c ca --test | tee ./log/test/DeepONet_nsch_ca.out
CUDA_VISIBLE_DEVICES=0 python train.py ./config/DeepONet/config_NSCH_ALL.yaml -c mob --test | tee ./log/test/DeepONet_nsch_mob.out
CUDA_VISIBLE_DEVICES=0 python train.py ./config/DeepONet/config_NSCH_ALL.yaml -c phi --test | tee ./log/test/DeepONet_nsch_phi.out
CUDA_VISIBLE_DEVICES=0 python train.py ./config/DeepONet/config_NSCH_ALL.yaml -c re --test | tee ./log/test/DeepONet_nsch_re.out
CUDA_VISIBLE_DEVICES=0 python train.py ./config/DeepONet/config_NSCH_ALL.yaml -c eps --test | tee ./log/test/DeepONet_nsch_eps.out
CUDA_VISIBLE_DEVICES=0 python train.py ./config/DeepONet/config_NSCH_ALL.yaml -c ibc_phi_ca_mob_re_eps --test | tee ./log/test/DeepONet_ibc_phi_ca_mob_re_eps.out
CUDA_VISIBLE_DEVICES=0 python train.py ./config/DeepONet/config_NSCH_ALL.yaml -c ca_mob_re_eps --test | tee ./log/test/DeepONet_nsch_ca_mob_re_eps.out
CUDA_VISIBLE_DEVICES=0 python train.py ./config/DeepONet/config_NSCH_ALL.yaml -c phi_ibc --test | tee ./log/test/DeepONet_nsch_phi_ibc.out

# # Darcy
CUDA_VISIBLE_DEVICES=0 python train.py ./config/DeepONet/config_Darcy_PDEBench.yaml -c PDEBench --test | tee ./log/test/DeepONet_darcy_PDEBench.out
CUDA_VISIBLE_DEVICES=0 python train.py ./config/DeepONet/config_Darcy_darcy.yaml -c darcy --test | tee ./log/test/DeepONet_darcy_darcy.out

# cylinder / ircylinder
CUDA_VISIBLE_DEVICES=0 python train.py ./config/DeepONet/config_ircylinder_irRE_irBC.yaml -c irRE --test | tee ./log/test/DeepONet_ircylinder_irRE.out
CUDA_VISIBLE_DEVICES=0 python train.py ./config/DeepONet/config_ircylinder_irRE_irBC.yaml -c irBC --test | tee ./log/test/DeepONet_ircylinder_irBC.out
CUDA_VISIBLE_DEVICES=0 python train.py ./config/DeepONet/config_cylinder_rRE_rBC.yaml -c rRE --test | tee ./log/test/DeepONet_cylinder_rRE.out
CUDA_VISIBLE_DEVICES=0 python train.py ./config/DeepONet/config_cylinder_rRE_rBC.yaml -c rRE_rBC --test | tee ./log/test/DeepONet_cylinder_rRE_rBC.out
CUDA_VISIBLE_DEVICES=0 python train.py ./config/DeepONet/config_cylinder_rRE_rBC.yaml -c rBC --test | tee ./log/test/DeepONet_cylinder_rBC.out

# periodic hills
CUDA_VISIBLE_DEVICES=0 python train.py ./config/DeepONet/config_hills_rRE.yaml -c rRE --test | tee ./log/test/DeepONet_hills_rRE.out
CUDA_VISIBLE_DEVICES=0 python train.py ./config/DeepONet/config_irhills_irRE.yaml -c irRE --test | tee ./log/test/DeepONet_irhills_irRE.out