import hashlib
import random

# 生成大素数 p 和生成元 g, h
def generate_parameters():
    p = 23  # 选择一个小素数作为示例
    g = 5   # 生成元
    h = 7   # 另一个生成元
    return p, g, h

# Pedersen承诺
def pedersen_commitment(m, r, p, g, h):
    C = (pow(g, m, p) * pow(h, r, p)) % p
    return C

# Schnorr非交互式零知识证明
def schnorr_proof(m, r, p, g, h):
    # 生成随机数 k
    k = random.randint(1, p-1)
    R = pow(g, k, p)
    e = int(hashlib.sha256(f'{R}'.encode()).hexdigest(), 16) % p
    s = (k + e * m) % p
    return (R, s, e)

# 验证Schnorr非交互式零知识证明
def verify_schnorr(C, R, s, e, p, g, h):
    lhs = (pow(g, s, p) * pow(h, e, p)) % p
    rhs = (R * pow(C, e, p)) % p
    return lhs == rhs

# 设置参数
p, g, h = generate_parameters()

# 生成消息 m 和随机数 r
m = 10
r = random.randint(1, p-1)

# 生成承诺
C = pedersen_commitment(m, r, p, g, h)
print(f'Pedersen承诺: C = {C}')

# 生成非交互式零知识证明
R, s, e = schnorr_proof(m, r, p, g, h)
print(f'Schnorr NIZK证明: (R, s, e) = ({R}, {s}, {e})')

# 验证非交互式零知识证明
is_valid = verify_schnorr(C, R, s, e, p, g, h)
print(f'证明验证结果: {is_valid}')
