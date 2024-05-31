# tube
CUDA_VISIBLE_DEVICES=0 python train.py ./config/geoFNO/config_tube_prop_bc_geo.yaml -c geo --test | tee ./log/test/geoFNO_tube_geo.out
CUDA_VISIBLE_DEVICES=0 python train.py ./config/geoFNO/config_tube_prop_bc_geo.yaml -c bc --test | tee ./log/test/geoFNO_tube_bc.out
CUDA_VISIBLE_DEVICES=0 python train.py ./config/geoFNO/config_tube_prop_bc_geo.yaml -c prop --test | tee ./log/test/geoFNO_tube_prop.out
CUDA_VISIBLE_DEVICES=0 python train.py ./config/geoFNO/config_tube_prop_bc_geo.yaml -c bc_geo --test | tee ./log/test/geoFNO_tube_bc_geo.out
CUDA_VISIBLE_DEVICES=0 python train.py ./config/geoFNO/config_tube_prop_bc_geo.yaml -c prop_geo --test | tee ./log/test/geoFNO_tube_prop_geo.out
CUDA_VISIBLE_DEVICES=0 python train.py ./config/geoFNO/config_tube_prop_bc_geo.yaml -c prop_bc --test | tee ./log/test/geoFNO_tube_prop_bc.out
CUDA_VISIBLE_DEVICES=0 python train.py ./config/geoFNO/config_tube_prop_bc_geo.yaml -c prop_bc_geo --test | tee ./log/test/geoFNO_tube_prop_bc_geo.out

# TGV
CUDA_VISIBLE_DEVICES=0 python train.py ./config/geoFNO/config_TGV_all.yaml -c all --test | tee ./log/test/geoFNO_tgv_all.out
CUDA_VISIBLE_DEVICES=0 python train.py ./config/geoFNO/config_TGV_all.yaml -c single --test | tee ./log/test/geoFNO_tgv_single.out

# cavity
CUDA_VISIBLE_DEVICES=0 python train.py ./config/geoFNO/config_cavity_ReD_bc_re.yaml -c ReD --test | tee ./log/test/geoFNO_cavity_ReD.out
CUDA_VISIBLE_DEVICES=0 python train.py ./config/geoFNO/config_cavity_ReD_bc_re.yaml -c bc --test | tee ./log/test/geoFNO_cavity_bc.out
CUDA_VISIBLE_DEVICES=0 python train.py ./config/geoFNO/config_cavity_ReD_bc_re.yaml -c re --test | tee ./log/test/geoFNO_cavity_re.out
CUDA_VISIBLE_DEVICES=0 python train.py ./config/geoFNO/config_cavity_ReD_bc_re.yaml -c ReD_bc_re --test | tee ./log/test/geoFNO_cavity_ReD_bc_re.out

# NSCH
CUDA_VISIBLE_DEVICES=0 python train.py ./config/geoFNO/config_NSCH_ALL.yaml -c bc --test | tee ./log/test/geoFNO_nsch_bc.out
CUDA_VISIBLE_DEVICES=0 python train.py ./config/geoFNO/config_NSCH_ALL.yaml -c ca --test | tee ./log/test/geoFNO_nsch_ca.out
CUDA_VISIBLE_DEVICES=0 python train.py ./config/geoFNO/config_NSCH_ALL.yaml -c mob --test | tee ./log/test/geoFNO_nsch_mob.out
CUDA_VISIBLE_DEVICES=0 python train.py ./config/geoFNO/config_NSCH_ALL.yaml -c phi --test | tee ./log/test/geoFNO_nsch_phi.out
CUDA_VISIBLE_DEVICES=0 python train.py ./config/geoFNO/config_NSCH_ALL.yaml -c pre --test | tee ./log/test/geoFNO_nsch_pre.out
CUDA_VISIBLE_DEVICES=0 python train.py ./config/geoFNO/config_NSCH_ALL.yaml -c re --test | tee ./log/test/geoFNO_nsch_re.out
CUDA_VISIBLE_DEVICES=0 python train.py ./config/geoFNO/config_NSCH_ALL.yaml -c uv0 --test | tee ./log/test/geoFNO_nsch_uv0.out
CUDA_VISIBLE_DEVICES=0 python train.py ./config/geoFNO/config_NSCH_ALL.yaml -c bc_ca_mob_phi_pre_re_uv0 --test | tee ./log/test/geoFNO_nsch_bc_ca_mob_phi_pre_re_uv0.out
CUDA_VISIBLE_DEVICES=0 python train.py ./config/geoFNO/config_NSCH_ALL.yaml -c ca_mob_re --test | tee ./log/test/geoFNO_nsch_ca_mob_re.out
CUDA_VISIBLE_DEVICES=0 python train.py ./config/geoFNO/config_NSCH_ALL.yaml -c phi_pre_uv0 --test | tee ./log/test/geoFNO_nsch_phi_pre_uv0.out

# Darcy
CUDA_VISIBLE_DEVICES=0 python train.py ./config/geoFNO/config_Darcy_PDEBench.yaml -c PDEBench --test | tee ./log/test/geoFNO_darcy_PDEBench.out

# cylinder / ircylinder
CUDA_VISIBLE_DEVICES=0 python train.py ./config/geoFNO/config_ircylinder_irRE_irBC.yaml -c irRE --test | tee ./log/test/geoFNO_ircylinder_irRE.out
CUDA_VISIBLE_DEVICES=0 python train.py ./config/geoFNO/config_ircylinder_irRE_irBC.yaml -c irBC --test | tee ./log/test/geoFNO_ircylinder_irBC.out
CUDA_VISIBLE_DEVICES=0 python train.py ./config/geoFNO/config_cylinder_rRE_rBC.yaml -c rRE_rBC --test | tee ./log/test/geoFNO_cylinder_rRE_rBC.out
CUDA_VISIBLE_DEVICES=0 python train.py ./config/geoFNO/config_cylinder_rRE_rBC.yaml -c rRE --test | tee ./log/test/geoFNO_cylinder_rRE.out
CUDA_VISIBLE_DEVICES=0 python train.py ./config/geoFNO/config_cylinder_rRE_rBC.yaml -c rBC --test | tee ./log/test/geoFNO_cylinder_rBC.out