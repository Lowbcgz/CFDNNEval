### extrapolation
# CUDA_VISIBLE_DEVICES=0 python extrapolation_test.py ./config/DeepONet/config_cylinder_rRE_rBC.yaml -c rRE | tee ./log/extp/DeepONet_extrapolation_rRE.out
# CUDA_VISIBLE_DEVICES=0 python extrapolation_test.py ./config/DeepONet/config_ircylinder_irRE_irBC.yaml -c irRE | tee ./log/extp/DeepONet_extrapolation_irRE.out

### plotting

# #  tube
# CUDA_VISIBLE_DEVICES=0 python transfering_test.py ./config/DeepONet/config_tube_prop_bc_geo.yaml -cs prop_bc_geo -ct bc,geo,prop

# # TGV
# CUDA_VISIBLE_DEVICES=0 python transfering_test.py ./config/DeepONet/config_TGV_all.yaml -cs all -ct single

# # cavity
# CUDA_VISIBLE_DEVICES=0 python transfering_test.py ./config/DeepONet/config_cavity_ReD_bc_re.yaml -cs ReD_bc_re -ct ReD,bc,re

# # NSCH
# CUDA_VISIBLE_DEVICES=0 python transfering_test.py ./config/DeepONet/config_NSCH_ALL.yaml -cs ibc_phi_ca_mob_re_eps -ct ibc,phi,ca,mob,re,eps
# CUDA_VISIBLE_DEVICES=0 python transfering_test.py ./config/DeepONet/config_NSCH_ALL.yaml -cs ca_mob_re_eps -ct ca,mob,re,eps
# CUDA_VISIBLE_DEVICES=0 python transfering_test.py ./config/DeepONet/config_NSCH_ALL.yaml -cs phi_ibc -ct ibc,phi

# # Cylinder
# CUDA_VISIBLE_DEVICES=0 python transfering_test.py ./config/DeepONet/config_cylinder_rRE_rBC.yaml -cs rRE_rBC -ct rRE,rBC


# collect metrics like nmse

# NSCH
CUDA_VISIBLE_DEVICES=0 python collect_nmse.py ./config/DeepONet/config_NSCH_ALL.yaml -c eps

# TGV
CUDA_VISIBLE_DEVICES=0 python collect_nmse.py ./config/DeepONet/config_TGV_all.yaml -c Re_ReD

# cavity
CUDA_VISIBLE_DEVICES=0 python collect_nmse.py ./config/DeepONet/config_cavity_ReD_bc_re.yaml -c ReD

# tube
CUDA_VISIBLE_DEVICES=0 python collect_nmse.py ./config/DeepONet/config_tube_prop_bc_geo.yaml -c bc

# cylinder
CUDA_VISIBLE_DEVICES=0 python collect_nmse.py ./config/DeepONet/config_cylinder_rRE_rBC.yaml -c rRE
CUDA_VISIBLE_DEVICES=0 python collect_nmse.py ./config/DeepONet/config_ircylinder_irRE_irBC.yaml -c irRE

# periodic hills
CUDA_VISIBLE_DEVICES=0 python collect_nmse.py ./config/DeepONet/config_hills_rRE.yaml -c rRE
CUDA_VISIBLE_DEVICES=0 python collect_nmse.py ./config/DeepONet/config_irhills_irRE.yaml -c irRE