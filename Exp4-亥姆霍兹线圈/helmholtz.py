import numpy as np
import matplotlib.pyplot as plt

# --- 物理和线圈参数 ---
MU0 = 4 * np.pi * 1e-7  # 真空磁导率 (T*m/A)
I = 1.0  # 电流 (A)

def Helmholtz_coils(r_low, r_up, d):
    '''
    计算双线圈系统的磁场
    '''
    print(f"开始计算磁场: r_low={r_low}, r_up={r_up}, d={d}")

    # 定义积分角度和空间网格
    phi_angles = np.linspace(0, 2*np.pi, 100)
    max_r = max(r_low, r_up)
    y_coords = np.linspace(-2*max_r, 2*max_r, 50)
    z_coords = np.linspace(-1.5*d, 1.5*d, 50)
    
    # 创建三维网格
    Y, Z, Phi = np.meshgrid(y_coords, z_coords, phi_angles, indexing='ij')

    # 计算到下方线圈的距离
    dist1_sq = (r_low * np.cos(Phi))**2 + (Y - r_low * np.sin(Phi))**2 + (Z + d/2)**2
    dist1 = np.sqrt(dist1_sq)
    dist1[dist1 < 1e-9] = 1e-9

    # 计算到上方线圈的距离
    dist2_sq = (r_up * np.cos(Phi))**2 + (Y - r_up * np.sin(Phi))**2 + (Z - d/2)**2
    dist2 = np.sqrt(dist2_sq)
    dist2[dist2 < 1e-9] = 1e-9

    # 计算磁场贡献的被积函数
    dBy_integrand = r_low * (Z + d/2) * np.sin(Phi) / dist1**3 + \
                    r_up * (Z - d/2) * np.sin(Phi) / dist2**3
    
    dBz_integrand = r_low * (r_low - Y * np.sin(Phi)) / dist1**3 + \
                    r_up * (r_up - Y * np.sin(Phi)) / dist2**3

    # 数值积分
    By_unscaled = np.trapz(dBy_integrand, x=phi_angles, axis=-1)
    Bz_unscaled = np.trapz(dBz_integrand, x=phi_angles, axis=-1)

    # 引入物理常数因子
    scaling_factor = (MU0 * I) / (4 * np.pi)
    By = scaling_factor * By_unscaled
    Bz = scaling_factor * Bz_unscaled
    
    print("磁场计算完成.")
    return Y[:,:,0], Z[:,:,0], By, Bz

def plot_magnetic_field_streamplot(r_coil_1, r_coil_2, d_coils):
    """
    绘制磁场流线图
    """
    print(f"开始绘图准备: r_coil_1={r_coil_1}, r_coil_2={r_coil_2}, d_coils={d_coils}")
    
    Y_plot, Z_plot, By_field, Bz_field = Helmholtz_coils(r_coil_1, r_coil_2, d_coils)

    plt.figure(figsize=(8, 7))
    
    # 计算磁场强度用于着色
    B_magnitude = np.sqrt(By_field**2 + Bz_field**2)
    
    # 绘制流线图
    stream = plt.streamplot(Y_plot, Z_plot, By_field, Bz_field,
                          color=B_magnitude, cmap='viridis',
                          linewidth=1.5, density=1.5,
                          arrowstyle='->', arrowsize=1.0)
    
    # 添加颜色条
    cbar = plt.colorbar(stream.lines)
    cbar.set_label('Magnetic Field Strength (T)')

    # 绘制线圈位置
    plt.plot([-r_coil_1, r_coil_1], [-d_coils/2, -d_coils/2], 'b-', linewidth=3, label='Coil 1')
    plt.plot([-r_coil_2, r_coil_2], [d_coils/2, d_coils/2], 'r-', linewidth=3, label='Coil 2')
    
    # 设置图形属性
    plt.xlabel('y (m)')
    plt.ylabel('z (m)')
    title = 'Standard Helmholtz Coils' if (r_coil_1 == r_coil_2 and d_coils == r_coil_1) else 'Two Coil System'
    plt.title(f'{title}\n(R1={r_coil_1}m, R2={r_coil_2}m, d={d_coils}m)')
    plt.gca().set_aspect('equal')
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend()

    plt.tight_layout()
    plt.show()
    print("绘图完成.")

# --- 主程序 ---
if __name__ == "__main__":
    # 标准亥姆霍兹线圈配置
    radius_1 = 0.5  # 下方线圈半径 (m)
    radius_2 = 0.5  # 上方线圈半径 (m)
    distance_between_coils = 0.5  # 两线圈中心距离 (m)

    plot_magnetic_field_streamplot(radius_1, radius_2, distance_between_coils)

    # 非标准配置示例（取消注释以测试）
    # print("\nTesting non-Helmholtz configuration:")
    # plot_magnetic_field_streamplot(0.5, 0.5, 0.8)
    # plot_magnetic_field_streamplot(0.3, 0.7, 0.6)
