import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import dblquad
from numpy.polynomial.legendre import leggauss

# 物理常数
G = 6.67430e-11  # 万有引力常数 (m^3 kg^-1 s^-2)

def calculate_sigma(length, mass):
    """
    计算薄片的面密度
    """
    return mass / (length ** 2)

def integrand(x, y, z):
    """
    引力积分核函数
    """
    r_squared = x**2 + y**2 + z**2
    return z / (r_squared ** 1.5)

def gauss_legendre_integral(length, z, n_points=100):
    """
    高斯-勒让德求积法计算二重积分
    """
    # 获取高斯点和权重
    points, weights = leggauss(n_points)
    
    # 变换积分区间
    a = -length/2
    b = length/2
    scale = (b - a)/2
    offset = (b + a)/2
    
    # 双重积分计算
    integral = 0.0
    for i in range(n_points):
        x = scale * points[i] + offset
        wx = weights[i]
        for j in range(n_points):
            y = scale * points[j] + offset
            wy = weights[j]
            integral += wx * wy * integrand(x, y, z)
    
    return integral * scale**2

def calculate_force(length, mass, z, method='gauss'):
    """
    计算给定高度处的引力
    """
    sigma = calculate_sigma(length, mass)
    if method == 'gauss':
        integral = gauss_legendre_integral(length, z)
    else:  # scipy
        integral, _ = dblquad(lambda x, y: integrand(x, y, z),
                            -length/2, length/2,
                            lambda x: -length/2, lambda x: length/2)
    
    return G * sigma * integral

def plot_force_vs_height(length, mass, z_min=0.1, z_max=10, n_points=100):
    """
    绘制引力随高度变化的曲线
    """
    # 生成高度点
    z_values = np.linspace(z_min, z_max, n_points)
    
    # 计算引力值
    F_gauss = [calculate_force(length, mass, z, 'gauss') for z in z_values]
    F_scipy = [calculate_force(length, mass, z, 'scipy') for z in z_values]
    
    # 绘制曲线
    plt.figure(figsize=(10, 6))
    plt.plot(z_values, F_gauss, 'b-', label='Gauss-Legendre (n=100)')
    plt.plot(z_values, F_scipy, 'r--', label='SciPy dblquad')
    
    # 添加理论极限线
    plt.axhline(y=2*np.pi*G*calculate_sigma(length, mass), 
               color='k', linestyle=':', label='Infinite plate limit')
    
    # 设置图表属性
    plt.title(f'Gravitational Force by a Square Plate (L={length}m, M={mass/1000:.0f} tons)')
    plt.xlabel('Height z (m)')
    plt.ylabel('Vertical Force F_z (N)')
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend()
    plt.yscale('log')
    plt.xscale('log')
    plt.tight_layout()
    plt.show()

# 主程序
if __name__ == '__main__':
    # 参数设置
    length = 10  # 边长 (m)
    mass = 1e4   # 质量 (kg)
    
    # 计算并绘制引力曲线
    plot_force_vs_height(length, mass)
    
    # 打印关键点引力值
    print("关键高度点的引力值比较:")
    print("-" * 50)
    print("高度(m)\t高斯积分(N)\tSciPy积分(N)")
    print("-" * 50)
    for z in [0.1, 0.5, 1, 5, 10]:
        F_g = calculate_force(length, mass, z, 'gauss')
        F_s = calculate_force(length, mass, z, 'scipy')
        print(f"{z:.1f}\t{F_g:.3e}\t{F_s:.3e}")
