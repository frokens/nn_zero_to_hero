import torch
import torch.nn as nn
from torchdiffeq import odeint

# 定义 ODE 网络模块
class ODEFunc(nn.Module):
    def __init__(self, dim):
        super(ODEFunc, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, 50),
            nn.Tanh(),
            nn.Linear(50, dim)
        )KOKO
    
    def forward(self, t, y):
        return self.net(y)

# 定义 Neural ODE 模型
class NeuralODE(nn.Module):
    def __init__(self, ode_func, time_span):
        super(NeuralODE, self).__init__()
        self.ode_func = ode_func
        self.integration_time = torch.tensor(time_span).float()
        
    def forward(self, y0):
        # 求解 ODE，输出在 t_end 时刻的状态
        y = odeint(self.ode_func, y0, self.integration_time)
        return y[-1]
    
    
# 示例：使用 2 维输入
dim = 2
ode_func = ODEFunc(dim)
model = NeuralODE(ode_func, time_span=[0, 1.0])

# 假设输入初始状态
y0 = torch.tensor([[1.0, 0.0]])
# 模型前向传播得到 t=1 时刻的输出
output = model(y0)

print("Neural ODE 输出：", output)
