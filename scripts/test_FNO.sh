# tube
# CUDA_VISIBLE_DEVICES=0 python train.py ./config/FNO/config_tube_prop_bc_geo.yaml -c geo --test | tee ./log/test/tube_geo.out
# CUDA_VISIBLE_DEVICES=0 python train.py ./config/FNO/config_tube_prop_bc_geo.yaml -c bc --test | tee ./log/test/tube_bc.out
# CUDA_VISIBLE_DEVICES=0 python train.py ./config/FNO/config_tube_prop_bc_geo.yaml -c prop --test | tee ./log/test/tube_prop.out
# CUDA_VISIBLE_DEVICES=0 python train.py ./config/FNO/config_tube_prop_bc_geo.yaml -c bc_geo --test | tee ./log/test/tube_bc_geo.out
# CUDA_VISIBLE_DEVICES=0 python train.py ./config/FNO/config_tube_prop_bc_geo.yaml -c prop_geo --test | tee ./log/test/tube_prop_geo.out
# CUDA_VISIBLE_DEVICES=0 python train.py ./config/FNO/config_tube_prop_bc_geo.yaml -c prop_bc --test | tee ./log/test/tube_prop_bc.out
# CUDA_VISIBLE_DEVICES=0 python train.py ./config/FNO/config_tube_prop_bc_geo.yaml -c prop_bc_geo --test | tee ./log/test/tube_prop_bc_geo.out

# TGV
# CUDA_VISIBLE_DEVICES=0 python train.py ./config/FNO/config_TGV_ALL.yaml -c rho_V0_nu --test | tee ./log/test/tgv_rho_V0_nu.out
# CUDA_VISIBLE_DEVICES=0 python train.py ./config/FNO/config_TGV_ALL.yaml -c rho --test | tee ./log/test/tgv_rho.out
# CUDA_VISIBLE_DEVICES=0 python train.py ./config/FNO/config_TGV_ALL.yaml -c V0 --test | tee ./log/test/tgv_V0.out
# CUDA_VISIBLE_DEVICES=0 python train.py ./config/FNO/config_TGV_ALL.yaml -c nu --test | tee ./log/test/tgv_nu.out

# cavity
# CUDA_VISIBLE_DEVICES=0 python train.py ./config/FNO/config_cavity_ReD_bc_re.yaml -c ReD --test | tee ./log/test/cavity_ReD.out
# CUDA_VISIBLE_DEVICES=0 python train.py ./config/FNO/config_cavity_ReD_bc_re.yaml -c bc --test | tee ./log/test/cavity_bc.out
# CUDA_VISIBLE_DEVICES=0 python train.py ./config/FNO/config_cavity_ReD_bc_re.yaml -c re --test | tee ./log/test/cavity_re.out
# CUDA_VISIBLE_DEVICES=0 python train.py ./config/FNO/config_cavity_ReD_bc_re.yaml -c ReD_bc_re --test | tee ./log/test/cavity_ReD_bc_re.out

# NSCH
# CUDA_VISIBLE_DEVICES=0 python train.py ./config/FNO/config_NSCH_ALL.yaml -c bc --test | tee ./log/test/nsch_bc.out
# CUDA_VISIBLE_DEVICES=0 python train.py ./config/FNO/config_NSCH_ALL.yaml -c ca --test | tee ./log/test/nsch_ca.out
# CUDA_VISIBLE_DEVICES=0 python train.py ./config/FNO/config_NSCH_ALL.yaml -c mob --test | tee ./log/test/nsch_mob.out
# CUDA_VISIBLE_DEVICES=0 python train.py ./config/FNO/config_NSCH_ALL.yaml -c phi --test | tee ./log/test/nsch_phi.out
# CUDA_VISIBLE_DEVICES=0 python train.py ./config/FNO/config_NSCH_ALL.yaml -c pre --test | tee ./log/test/nsch_pre.out
# CUDA_VISIBLE_DEVICES=0 python train.py ./config/FNO/config_NSCH_ALL.yaml -c re --test | tee ./log/test/nsch_re.out
# CUDA_VISIBLE_DEVICES=0 python train.py ./config/FNO/config_NSCH_ALL.yaml -c uv0 --test | tee ./log/test/nsch_uv0.out
# CUDA_VISIBLE_DEVICES=0 python train.py ./config/FNO/config_NSCH_ALL.yaml -c bc_ca_mob_phi_pre_re_uv0 --test | tee ./log/test/nsch_bc_ca_mob_phi_pre_re_uv0.out
# CUDA_VISIBLE_DEVICES=0 python train.py ./config/FNO/config_NSCH_ALL.yaml -c ca_mob_re --test | tee ./log/test/nsch_ca_mob_re.out
# CUDA_VISIBLE_DEVICES=0 python train.py ./config/FNO/config_NSCH_ALL.yaml -c phi_pre_uv0 --test | tee ./log/test/nsch_phi_pre_uv0.out

# Darcy
# CUDA_VISIBLE_DEVICES=0 python train.py ./config/FNO/config_Darcy_ar_as.yaml -c ar --test | tee ./log/test/darcy_ar.out
# CUDA_VISIBLE_DEVICES=0 python train.py ./config/FNO/config_Darcy_ar_as.yaml -c as --test | tee ./log/test/darcy_as.out