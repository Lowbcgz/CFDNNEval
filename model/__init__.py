from .fno import FNO2d, FNO3d
from .LSM.lsm_2d import LSM_2d
from .LSM.lsm_3d import LSM_3d
from .LSM.lsm_Irregular_Geo import LSM_2d_ir
from .uno.uno import *
from .kno import *
from .unet import *
from .auto_deeponet import AutoDeepONet,AutoDeepONet_3d
from .geofno import geoFNO2d
from .oformer.oformer import Oformer

from .GFormer.libs_path import *
from .GFormer.libs import *
from .GFormer.libs.ns_lite import FourierTransformer2DLite, My_FourierTransformer2D, My_FourierTransformer3D, Darcy_FourierTransformer2D