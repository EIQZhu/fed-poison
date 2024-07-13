import torch
import numpy as np

class SimpleCNN(torch.nn.Module):
    """
    用于测试，只能用于MNIST数据集
    """

    def __init__(self, num_classes=10):
        super(SimpleCNN, self).__init__()
        self.conv1 = torch.nn.Conv2d(1, 4, (3, 3))  # 4 * 13 * 13
        self.conv2 = torch.nn.Conv2d(4, 16, (4, 4))  # 16 * 5 * 5
        self.conv3 = torch.nn.Conv2d(16, 32, (3, 3))  # 32 * 3 * 3
        self.fc = torch.nn.Linear(32 * 9, num_classes)

    def forward(self, x):
        x = torch.relu(torch.max_pool2d(self.conv1(x), 2))
        x = torch.relu(torch.max_pool2d(self.conv2(x), 2))
        x = torch.relu(self.conv3(x))
        return self.fc(x.view(x.size(0), -1))

model = SimpleCNN()
#
# # 获取模型的 state_dict
state_dict = model.state_dict()
# # 定义缩放因子
# scaling_factor = 2**16  # 16位定点数
#
# # 将模型参数转换为整数
# int_state_dict = {}
# for key, value in state_dict.items():
#     int_state_dict[key] = (value * scaling_factor).to(torch.int32)
#
# # 打印转换后的整数参数
# for key, value in int_state_dict.items():
#     print(f"Integer Parameter: {key}")
#     print(f"Shape: {value.shape}")
#     print(value)
#     print("\n")
#     break
# 训练过程中将整数参数转换回浮点数进行计算
# 注意：这种方法不完全是整数运算，只是在存储时使用整数，计算时仍然会转换回浮点数
# for key in int_state_dict.keys():
#     state_dict[key] = (int_state_dict[key].to(torch.float32) / scaling_factor)

# # 打印 state_dict 的内容
# print("State Dict Keys: ", state_dict.keys())
for key, value in state_dict.items():
    print(f"Parameter: {key}")
    print(f"Shape: {value.shape}")
    print(value)
    print("\n")
    break

# # 假设 prg_pairwise 是一个包含二进制数据的字典
# prg_pairwise = {}
# vector_dtype = np.int32
# for key, value in state_dict.items():
#     prg_pairwise[key] = np.random.bytes(value.numel()*4 )  # 每个参数的二进制数据长度为 numel * 4
#
# # 将 prg_pairwise 中的二进制数据转换为向量
# # vec_prg_pairwise = {}
# # for key in prg_pairwise.keys():
# #     # vec_prg_pairwise[key] = np.frombuffer(prg_pairwise[key], dtype=vector_dtype)
# #     # print(vec_prg_pairwise[key] )
#
# # 确保每个向量的长度正确
# for key in vec_prg_pairwise.keys():
#     if len(vec_prg_pairwise[key]) != state_dict[key].numel():
#         raise RuntimeError(f"Vector length error for key {key}")
#
# # 将转换后的向量应用到模型参数上
# for key in state_dict.keys():
#     state_dict[key] += torch.tensor(vec_prg_pairwise[key].reshape(state_dict[key].shape), dtype=state_dict[key].dtype)
# #
# # # 打印应用掩码后的参数
# # for key, value in state_dict.items():
# #     print(f"Updated Parameter: {key}")
# #     print(f"Shape: {value.shape}")
# #     print(value)
#     print("\n")
#     break


