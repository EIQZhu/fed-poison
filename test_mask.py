import torch
import numpy as np

# 假设 state_dict 是你的模型的 state_dict
state_dict = {
    'param1': torch.randn(4, 1, 3, 3),
    'param2': torch.randn(10, 5),
    # 添加更多的参数
}

# 假设 avg_mask 是你要应用的掩码
avg_mask = np.random.randn(sum(param.numel() for param in state_dict.values()))

# 将所有参数展平并合并到一个向量中
flattened_params = []
for key in state_dict.keys():
    param = state_dict[key]
    print(f"{key}: {state_dict[key]}")
    flattened_params.append(param.flatten().numpy())

flattened_params = np.concatenate(flattened_params)
print(flattened_params)
# 确保 avg_mask 的长度和 flattened_params 一致
assert len(flattened_params) == len(avg_mask), "avg_mask 的长度应与所有参数展平后的长度一致"

# 将 avg_mask 应用到展平的参数向量
#flattened_params += avg_mask

# 将展平后的参数重新分配回原始的形状
start_idx = 0
for key in state_dict.keys():
    param_shape = state_dict[key].shape
    param_len = state_dict[key].numel()
    reshaped_param = torch.tensor(flattened_params[start_idx:start_idx + param_len].reshape(param_shape), dtype=state_dict[key].dtype)
    state_dict[key] = reshaped_param
    start_idx += param_len

# 打印结果以验证
for key in state_dict.keys():
    print(f"{key}: {state_dict[key]}")
