import numpy as np
import h5py

def create_label(x_min, x_max, y_min, y_max, t_min, t_max, v, rho=1, L=1):
    axis_x = np.linspace(x_min, x_max, num=64, endpoint=True)
    axis_y = np.linspace(y_min, y_max, num=64, endpoint=True)
    axis_t = np.linspace(t_min, t_max, num=101, endpoint=True)

    # 生成三维网格
    mesh_x, mesh_t, mesh_y = np.meshgrid(axis_x, axis_t, axis_y)
    inputs = np.stack((mesh_x, mesh_y, mesh_t), axis=-1).reshape(-1, 3).astype(np.float32)

    num_points = inputs.shape[0]  # 128*128*101个时空点
    label = np.zeros((num_points, 3), dtype=np.float32)
    
    for i in range(num_points):
        x, y, t = inputs[i]
        u_x = -np.cos(x / L) * np.sin(y / L) * np.exp(-2 * v * t / L**2)
        u_y = np.sin(x / L) * np.cos(y / L) * np.exp(-2 * v * t / L**2)
        p = -rho / 4 * (np.cos(2 * x / L) + np.cos(2 * y / L)) * np.exp(-4 * v * t / L**2)
        label[i] = np.float32([u_x, u_y, p])

    inputs = inputs.reshape((101, 64, 64, 3))
    label = label.reshape((101, 64, 64, 3))

    # return: (101, 64, 64, 3)
    return label

# 时间范围设置
t_min = 0
t_max = 10

# 工况参数设置
# nu超过2下降太快，很早就超出机器精度了
# 1/100是个合适的间隔
v_range = np.arange(1/500, 2+1/500, 1/500)  # [0.01, 0.02, ..., 10.00] -- Re[100, ..., 0.1]

# 几何参数设置
edge_range = np.arange(1, 6) * np.pi  # [π, 2π, 3π, 4π, 5π]

for edge_long in edge_range:
    edge_name = int(edge_long / np.pi)  # For file naming
    file_name = f'./data_{edge_name}pi.h5'

    nu = np.zeros(v_range.size)
    uvp = np.zeros([v_range.size, 101, 64, 64, 3])  # Fixed to 101 for time dimension
    edge = np.ones(v_range.size) * edge_long

    with h5py.File(file_name, 'w') as f:
        for i in range(v_range.size):
            v = v_range[i]
            # 生成数据
            x_min = 0
            x_max = edge_long
            y_min = 0
            y_max = edge_long

            labels = create_label(x_min, x_max, y_min, y_max, t_min, t_max, v)
            uvp[i] = labels
            nu[i] = v

            if (i + 1) % 100 == 0:
                print(f'已经生成{i + 1}条数据')
        
        # 计算Re (Reynolds number)
        Re = np.divide(1, nu, out=np.zeros_like(nu), where=(nu != 0))  # Avoid division by zero

        # 创建数据集
        dset1 = f.create_dataset('uvp', data=uvp, dtype='f4', compression='gzip', compression_opts=4)
        dset2 = f.create_dataset('nu', data=nu, dtype='f4', compression='gzip', compression_opts=4)
        dset3 = f.create_dataset('edge', data=edge, dtype='f4', compression='gzip', compression_opts=4)
        dset4 = f.create_dataset('Re', data=Re, dtype='f4', compression='gzip', compression_opts=4)
        print(f'edge:{edge[0]}')
