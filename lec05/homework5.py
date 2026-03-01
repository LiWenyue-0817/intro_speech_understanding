import numpy as np
import matplotlib.pyplot as plt

def center_of_gravity(x):
    '''
    Find the center of gravity of a vector, x.
    If x=[x0,x1,...,xn], then you should return
    c = ( 0*x0 + 1*x1 + 2*x2 + ... + n*xn ) / sum(x)
    where n = len(x)-1.
    
    Recommended method: use np.arange, np.dot, and np.sum.
    
    @param:
    x (array): a 1d numpy array
    
    @result:
    c (scalar): x's center of gravity
    '''
    # 生成索引数组 [0,1,2,...,len(x)-1]
    indices = np.arange(len(x))
    # 计算分子：索引和对应元素的点积
    numerator = np.dot(indices, x)
    # 计算分母：数组元素和
    denominator = np.sum(x)
    # 避免除以0
    if denominator == 0:
        c = 0
    else:
        c = numerator / denominator
    return c

def matched_identity(x):
    '''
    Create an identity matrix that has the same number of rows as x has elements.
    Hint: use len(x), and use np.eye.
    
    @param:
    x (array): a 1d numpy array, of length N
    
    @result:
    I (array): a 2d numpy array: an NxN identity matrix
    '''
    # 创建NxN单位矩阵（N为x的长度）
    I = np.eye(len(x))
    return I

def sine_and_cosine(t_start, t_end, t_steps):
    '''
    Create a time axis, and compute its cosine and sine.
    Hint: use np.linspace, np.cos, and np.sin
    
    @param:
    t_start (scalar): the starting time
    t_end (scalar): the ending time
    t_steps (scalar): length of t, x, and y
    
    @result:
    t (array of length t_steps): time axis, t_start through t_end inclusive
    x (array of length t_steps): cos(t)
    y (array of length t_steps): sin(t)
    '''
    # 生成时间轴：从t_start到t_end，共t_steps个点
    t = np.linspace(t_start, t_end, t_steps)
    # 计算余弦值
    x = np.cos(t)
    # 计算正弦值
    y = np.sin(t)
    return t, x, y

# ------------------- 测试代码 -------------------
if __name__ == "__main__":
    # 测试center_of_gravity函数
    test_x = np.array([1, 2, 3, 4])
    cog = center_of_gravity(test_x)
    print(f"测试center_of_gravity函数：")
    print(f"输入数组: {test_x}")
    print(f"重心位置: {cog}\n")

    # 测试matched_identity函数
    identity_mat = matched_identity(test_x)
    print(f"测试matched_identity函数：")
    print(f"输入数组长度: {len(test_x)}")
    print(f"生成的单位矩阵:\n{identity_mat}\n")

    # 测试sine_and_cosine函数
    t, x_cos, y_sin = sine_and_cosine(0, 2*np.pi, 100)
    print(f"测试sine_and_cosine函数：")
    print(f"时间轴长度: {len(t)}")
    print(f"前5个时间点: {t[:5]}")
    print(f"前5个余弦值: {x_cos[:5]}")
    print(f"前5个正弦值: {y_sin[:5]}\n")

    # 可视化正弦和余弦曲线
    plt.figure(figsize=(8, 4))
    plt.plot(t, x_cos, label='cos(t)', color='blue')
    plt.plot(t, y_sin, label='sin(t)', color='red')
    plt.xlabel('t')
    plt.ylabel('Value')
    plt.title('Cosine and Sine Curves (0 to 2π)')
    plt.legend()
    plt.grid(True)
    plt.show()
