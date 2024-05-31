# # tube
# CUDA_VISIBLE_DEVICES=7 python train.py ./config/OFormer/config_tube_prop_bc_geo.yaml -c geo --test | tee ./log/OFormer/test/OFormer_tube_geo.out
# CUDA_VISIBLE_DEVICES=7 python train.py ./config/OFormer/config_tube_prop_bc_geo.yaml -c bc --test | tee ./log/OFormer/test/OFormer_tube_bc.out
# CUDA_VISIBLE_DEVICES=7 python train.py ./config/OFormer/config_tube_prop_bc_geo.yaml -c prop --test | tee ./log/OFormer/test/OFormer_tube_prop.out
# CUDA_VISIBLE_DEVICES=7 python train.py ./config/OFormer/config_tube_prop_bc_geo.yaml -c bc_geo --test | tee ./log/OFormer/test/OFormer_tube_bc_geo.out
# CUDA_VISIBLE_DEVICES=7 python train.py ./config/OFormer/config_tube_prop_bc_geo.yaml -c prop_geo --test | tee ./log/OFormer/test/OFormer_tube_prop_geo.out
# CUDA_VISIBLE_DEVICES=7 python train.py ./config/OFormer/config_tube_prop_bc_geo.yaml -c prop_bc --test | tee ./log/OFormer/test/OFormer_tube_prop_bc.out
# CUDA_VISIBLE_DEVICES=7 python train.py ./config/OFormer/config_tube_prop_bc_geo.yaml -c prop_bc_geo --test | tee ./log/OFormer/test/OFormer_tube_prop_bc_geo.out

# # TGV
# CUDA_VISIBLE_DEVICES=7 python train.py ./config/OFormer/config_TGV_all.yaml -c all --test | tee ./log/OFormer/test/OFormer_tgv_all.out
# CUDA_VISIBLE_DEVICES=7 python train.py ./config/OFormer/config_TGV_all.yaml -c single --test | tee ./log/OFormer/test/OFormer_tgv_single.out

# # cavity
# CUDA_VISIBLE_DEVICES=7 python train.py ./config/OFormer/config_cavity_ReD_bc_re.yaml -c ReD --test | tee ./log/OFormer/test/OFormer_cavity_ReD.out
# CUDA_VISIBLE_DEVICES=7 python train.py ./config/OFormer/config_cavity_ReD_bc_re.yaml -c bc --test | tee ./log/OFormer/test/OFormer_cavity_bc.out
# CUDA_VISIBLE_DEVICES=7 python train.py ./config/OFormer/config_cavity_ReD_bc_re.yaml -c re --test | tee ./log/OFormer/test/OFormer_cavity_re.out
CUDA_VISIBLE_DEVICES=7 python train.py ./config/OFormer/config_cavity_ReD_bc_re.yaml -c ReD_bc_re --test | tee ./log/OFormer/test/OFormer_cavity_ReD_bc_re.out

# # NSCH
# CUDA_VISIBLE_DEVICES=7 python train.py ./config/OFormer/config_NSCH_ALL.yaml -c ibc --test | tee ./log/OFormer/test/OFormer_nsch_ibc.out
# CUDA_VISIBLE_DEVICES=7 python train.py ./config/OFormer/config_NSCH_ALL.yaml -c ca --test | tee ./log/OFormer/test/OFormer_nsch_ca.out
# CUDA_VISIBLE_DEVICES=7 python train.py ./config/OFormer/config_NSCH_ALL.yaml -c mob --test | tee ./log/OFormer/test/OFormer_nsch_mob.out
# CUDA_VISIBLE_DEVICES=7 python train.py ./config/OFormer/config_NSCH_ALL.yaml -c phi --test | tee ./log/OFormer/test/OFormer_nsch_phi.out
# CUDA_VISIBLE_DEVICES=7 python train.py ./config/OFormer/config_NSCH_ALL.yaml -c re --test | tee ./log/OFormer/test/OFormer_nsch_re.out
# CUDA_VISIBLE_DEVICES=7 python train.py ./config/OFormer/config_NSCH_ALL.yaml -c eps --test | tee ./log/OFormer/test/OFormer_nsch_eps.out
# CUDA_VISIBLE_DEVICES=7 python train.py ./config/OFormer/config_NSCH_ALL.yaml -c ibc_phi_ca_mob_re_eps --test | tee ./log/OFormer/test/OFormer_ibc_phi_ca_mob_re_eps.out
# CUDA_VISIBLE_DEVICES=7 python train.py ./config/OFormer/config_NSCH_ALL.yaml -c ca_mob_re_eps --test | tee ./log/OFormer/test/OFormer_nsch_ca_mob_re_eps.out
CUDA_VISIBLE_DEVICES=7 python train.py ./config/OFormer/config_NSCH_ALL.yaml -c phi_ibc --test | tee ./log/OFormer/test/OFormer_nsch_phi_ibc.out

# # Darcy
# CUDA_VISIBLE_DEVICES=7 python train.py ./config/OFormer/config_Darcy_ar_as.yaml -c ar --test | tee ./log/OFormer/test/OFormer_darcy_ar.out
# CUDA_VISIBLE_DEVICES=7 python train.py ./config/OFormer/config_Darcy_ar_as.yaml -c as --test | tee ./log/OFormer/test/OFormer_darcy_as.out
# CUDA_VISIBLE_DEVICES=7 python train.py ./config/OFormer/config_Darcy_ar_as.yaml -c as_ar --test | tee ./log/OFormer/test/OFormer_darcy_as_ar.out
# CUDA_VISIBLE_DEVICES=7 python train.py ./config/OFormer/config_Darcy_PDEBench.yaml -c PDEBench --test | tee ./log/OFormer/test/OFormer_PDEBench.out


# # cylinder
# CUDA_VISIBLE_DEVICES=7 python train.py ./config/OFormer/config_cylinder_rRE_rBC.yaml -c rRE_rBC --test | tee ./log/OFormer/test/OFormer_cylinder_rRE_rBC.out
# CUDA_VISIBLE_DEVICES=7 python train.py ./config/OFormer/config_cylinder_rRE_rBC.yaml -c rRE --test | tee ./log/OFormer/test/OFormer_cylinder_rRE.out
# CUDA_VISIBLE_DEVICES=7 python train.py ./config/OFormer/config_cylinder_rRE_rBC.yaml -c rBC --test | tee ./log/OFormer/test/OFormer_cylinder_rBC.out
# CUDA_VISIBLE_DEVICES=7 python train.py ./config/OFormer/config_cylinder_irRE_irBC.yaml -c irRE --test | tee ./log/OFormer/test/OFormer_cylinder_irRE.out
# CUDA_VISIBLE_DEVICES=7 python train.py ./config/OFormer/config_cylinder_irRE_irBC.yaml -c irBC --test | tee ./log/OFormer/test/OFormer_cylinder_irBC.out