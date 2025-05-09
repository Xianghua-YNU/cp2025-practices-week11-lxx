import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import simpson

# --- 常量定义 ---
a = 1.0  # 圆环半径
C = 1.0 / (2 * np.pi)  # 常数因子 q/(2π)

# --- 计算函数 ---

def calculate_potential_on_grid(y_coords, z_coords):
    """
    在 yz 平面 (x=0) 的网格上计算电势 V(0, y, z)
    """
    print("开始计算电势...")
    
    # 创建网格和积分角度
    z_grid, y_grid = np.meshgrid(z_coords, y_coords)
    phi = np.linspace(0, 2*np.pi, 100)  # 积分角度
    
    # 计算场点到圆环上各点的距离
    # 圆环参数方程: (a*cosφ, a*sinφ, 0)
    # 场点: (0, y, z)
    R = np.sqrt((a*np.cos(phi)[:, np.newaxis, np.newaxis] - 0)**2 + 
                (a*np.sin(phi)[:, np.newaxis, np.newaxis] - y_grid[np.newaxis, :, :])**2 + 
                (0 - z_grid[np.newaxis, :, :])**2)
    
    # 处理R接近零的情况
    R[R < 1e-10] = 1e-10
    
    # 计算电势微元并积分
    dV = C / R
    V = simpson(dV, x=phi, axis=0)
    
    print("电势计算完成.")
    return V, y_grid, z_grid

def calculate_electric_field_on_grid(V, y_coords, z_coords):
    """
    根据电势 V 计算 yz 平面上的电场 E = -∇V
    """
    print("开始计算电场...")
    
    # 计算网格间距
    dy = y_coords[1] - y_coords[0]
    dz = z_coords[1] - z_coords[0]
    
    # 计算电势梯度
    grad_y, grad_z = np.gradient(-V, dy, dz)
    
    print("电场计算完成.")
    return grad_y, grad_z

# --- 可视化函数 ---

def plot_potential_and_field(y_coords, z_coords, V, Ey, Ez, y_grid, z_grid):
    """
    绘制 yz 平面上的等势线和电场线
    """
    print("开始绘图...")
    fig = plt.figure('Potential and Electric Field of Charged Ring (yz plane, x=0)', figsize=(12, 6))
    
    # 等势线图
    plt.subplot(1, 2, 1)
    levels = np.linspace(0.1, 1.2, 20)
    contour = plt.contourf(y_grid, z_grid, V, levels=levels, cmap='viridis')
    plt.colorbar(contour, label='Electric Potential (V)')
    plt.contour(y_grid, z_grid, V, levels=levels, colors='k', linewidths=0.5)
    plt.xlabel('y / a')
    plt.ylabel('z / a')
    plt.title('Electric Potential')
    plt.gca().set_aspect('equal')
    plt.grid(True)
    
    # 电场线图
    plt.subplot(1, 2, 2)
    E_magnitude = np.sqrt(Ey**2 + Ez**2)
    stream = plt.streamplot(y_grid, z_grid, Ey, Ez, color=E_magnitude, 
                          cmap='hot', linewidth=1, density=1.5, arrowstyle='->')
    plt.colorbar(stream.lines, label='Electric Field Magnitude')
    plt.plot([-a, a], [0, 0], 'ro', markersize=5, label='Charged Ring')
    plt.xlabel('y / a')
    plt.ylabel('z / a')
    plt.title('Electric Field Lines')
    plt.gca().set_aspect('equal')
    plt.grid(True)
    plt.legend()
    
    plt.tight_layout()
    plt.show()
    print("绘图完成.")

# --- 主程序 ---
if __name__ == "__main__":
    # 定义计算区域
    num_points = 50
    range_factor = 2
    y_range = np.linspace(-range_factor*a, range_factor*a, num_points)
    z_range = np.linspace(-range_factor*a, range_factor*a, num_points)
    
    # 计算电势和电场
    V, y_grid, z_grid = calculate_potential_on_grid(y_range, z_range)
    Ey, Ez = calculate_electric_field_on_grid(V, y_range, z_range)
    
    # 可视化
    plot_potential_and_field(y_range, z_range, V, Ey, Ez, y_grid, z_grid)
