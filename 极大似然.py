import numpy as np
from scipy.stats import norm


μ = 30  # 数学期望
σ = 2  # 方差

x=[1,2,2,3]

print(norm.fit(x))  # 返回极大似然估计，估计出参数约为30和2
