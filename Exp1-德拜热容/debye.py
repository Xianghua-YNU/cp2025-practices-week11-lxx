import numpy as np
import matplotlib.pyplot as plt
from numpy.polynomial.legendre import leggauss

# 物理常数
kB = 1.380649e-23  # 玻尔兹曼常数，单位：J/K

# 样本参数
V = 1000e-6  # 体积，1000立方厘米转换为立方米
rho = 6.022e28  # 原子数密度，单位：m^-3
theta_D = 428  # 德拜温度，单位：K

def integrand(x):
    """被积函数：x^4 * e^x / (e^x - 1)^2
    
    参数：
    x : float 或 numpy.ndarray
        积分变量
    
    返回：
    float 或 numpy.ndarray：被积函数的值
    """
    # 处理x=0的情况（虽然积分下限是0，但高斯点不会正好在0）
    mask = x < 1e-10
    result = np.zeros_like(x)
    
    # 对于x不为0的点
    x_nonzero = x[~mask] if isinstance(x, np.ndarray) else x
    if isinstance(x, np.ndarray) or x >= 1e-10:
        exp_x = np.exp(x_nonzero)
        numerator = x_nonzero**4 * exp_x
        denominator = (exp_x - 1)**2
        result_nonzero = numerator / denominator
        
        if isinstance(x, np.ndarray):
            result[~mask] = result_nonzero
        else:
            result = result_nonzero
    
    # 对于x接近0的点，使用泰勒展开近似
    if isinstance(x, np.ndarray):
        result[mask] = x[mask]**2  # 当x→0时，x^4 e^x/(e^x-1)^2 ≈ x^2
    elif x < 1e-10:
        result = x**2
    
    return result

def gauss_quadrature(f, a, b, n=50):
    """实现高斯-勒让德积分
    
    参数：
    f : callable
        被积函数
    a, b : float
        积分区间的端点
    n : int
        高斯点的数量
    
    返回：
    float：积分结果
    """
    # 获取高斯点和权重
    points, weights = leggauss(n)
    
    # 变换积分区间从[-1,1]到[a,b]
    transformed_points = 0.5*(b - a)*points + 0.5*(b + a)
    transformed_weights = 0.5*(b - a)*weights
    
    # 计算积分
    integral = np.sum(transformed_weights * f(transformed_points))
    return integral

def cv(T):
    """计算给定温度T下的热容
    
    参数：
    T : float
        温度，单位：K
    
    返回：
    float：热容值，单位：J/K
    """
    if T == 0:
        return 0.0
    
    # 积分上限
    upper_limit = theta_D / T
    
    # 计算积分
    integral = gauss_quadrature(integrand, 0, upper_limit)
    
    # 计算热容
    Cv = 9 * V * rho * kB * (T / theta_D)**3 * integral
    return Cv

def plot_cv():
    """绘制热容随温度的变化曲线"""
    # 生成温度范围
    temperatures = np.linspace(5, 500, 100)
    
    # 计算热容
    heat_capacities = np.array([cv(T) for T in temperatures])
    
    # 绘制曲线
    plt.figure(figsize=(10, 6))
    plt.plot(temperatures, heat_capacities, 'b-', linewidth=2)
    
    # 添加标记点
    test_points = [50, 100, 300, 428, 500]
    for T in test_points:
        Cv = cv(T)
        plt.plot(T, Cv, 'ro')
        plt.text(T, Cv, f'  {Cv:.2e} J/K', verticalalignment='bottom')
    
    # 添加标签和标题
    plt.xlabel('Temperature (K)', fontsize=12)
    plt.ylabel('Heat Capacity (J/K)', fontsize=12)
    plt.title('Debye Model: Heat Capacity of Aluminum vs Temperature', fontsize=14)
    plt.grid(True, linestyle='--', alpha=0.6)
    
    # 显示德拜温度线
    plt.axvline(x=theta_D, color='gray', linestyle='--', label=f'Debye Temperature ({theta_D} K)')
    plt.legend()
    
    plt.tight_layout()
    plt.show()

def test_cv():
    """测试热容计算函数"""
    # 测试一些特征温度点的热容值
    test_temperatures = [5, 50, 100, 300, 428, 500]
    print("\n测试不同温度下的热容值：")
    print("-" * 50)
    print("温度 (K)\t热容 (J/K)\tCv/(3NkB)")
    print("-" * 50)
    for T in test_temperatures:
        result = cv(T)
        N = V * rho  # 总原子数
        dulong_petit = 3 * N * kB  # 杜隆-珀替极限
        ratio = result / dulong_petit
        print(f"{T:8.1f}\t{result:12.3e}\t{ratio:.3f}")

def main():
    # 运行测试
    test_cv()
    
    # 绘制热容曲线
    plot_cv()

if __name__ == '__main__':
    main()
