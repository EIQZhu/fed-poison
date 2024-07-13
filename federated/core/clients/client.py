import pickle
import socket
import time
import struct

import torch
from torch.utils.data import DataLoader
#from util import param
import numpy as np

# pycryptodomex library functions
from Cryptodome.PublicKey import ECC
from Cryptodome.Cipher import AES, ChaCha20
from Cryptodome.Random import get_random_bytes
from Cryptodome.Hash import SHA256
from Cryptodome.Signature import DSS
import hashlib
from util import param
from util.crypto import ecchash
#from util.crypto.secretsharing import secret_int_to_points, points_to_secret_int


class BaseClient:
    def __init__(
            self,
            #idxx: int,
            idx:int,
            ip: str,
            port: int,
            server_ip: str,
            server_port: int,
            model: str,
            data: DataLoader,
            sample_num: int,
            n_classes: int,
            global_epoch: int,
            local_epoch: int,
            optimizer: str,
            lr: float,
            device: str,
            #n_clients:int,
            neighborhood_size:int,
            #key_length:int,
            criterion=torch.nn.CrossEntropyLoss(),
            #idx:int
            #prime : int
    ):
        self.vector_len=5000
        self.idx = idx
        self.model = None
        self.optimizer = None
        self.ip = ip
        self.port = port
        self.server_ip = server_ip
        self.server_port = server_port
        self.criterion = criterion  # 损失函数
        self.data = data  # 数据
        self.sample_num = sample_num
        self.device = torch.device(device)  # 设备
        self.lr = lr  # 学习率
        self.global_epoch = global_epoch
        self.local_epoch = local_epoch  # 本地多轮迭代次数
        self.loss = []  # 本地训练的损失
        # self.optimizer = torch.optim.SGD(self.model.parameters(), lr=0.01)
        self.model_name = model
        self.optim_name = optimizer
        self.n_classes = n_classes
        self.neighborhood_size = neighborhood_size
        self.neighbors_list = set()  # neighbors
        self.n_clients = 4 ##看能不能用serverd的配置文件？？？？？？
        self.vector_dtype = np.int32
        self.key_length = 32
        self.prime = ecchash.n


        """ Read keys. """
        # sk is used to establish pairwise secret with neighbors' public keys
        try:
            hdr = 'pki_files/client' + str(self.idx - 1) + '.pem'
            f = open(hdr, "rt")
            self.key = ECC.import_key(f.read())
            self.secret_key = self.key.d#自己的ai吗？
            f.close()
        except IOError:
            raise RuntimeError("No such file. Run setup_pki.py first.")

        print(f"CLIENT@{self.ip}:{self.port} INFO: Start!")

    def first_pull(self):#接收初始模型参数,有全局？没把
        client_socket = socket.socket()
        client_socket.bind((self.ip, self.port))
        client_socket.connect((self.server_ip, self.server_port))
        #self.client_recv(client_socket) 返回的是一个包含模型参数的字典
        self.model.load_state_dict(self.client_recv(client_socket))#第一次接收的模型？
        client_socket.setsockopt(socket.SOL_SOCKET, socket.SO_LINGER, struct.pack('ii', 1, 0))
        client_socket.close()

#main->config->run
    def run(self):
        from ..register import all_arch, all_optim
        self.model = all_arch[self.model_name](num_classes=self.n_classes).to(self.device)  # 模型
        self.optimizer = all_optim[self.optim_name](self.model.parameters(), lr=self.lr)
        self.first_pull()#recieve 初始模型
        for ge in range(self.global_epoch):
            for epoch in range(self.local_epoch):
                loss_avg = self.train()
                self.loss.append(loss_avg)
                print(
                    f"CLIENT@{self.ip}:{self.port} INFO: "
                    f"Global Epoch[{ge + 1}|{self.global_epoch}] "
                    f"Local Epoch[{epoch + 1}|{self.local_epoch}] "
                    f"Loss:{round(loss_avg, 3)}")
            #本地迭代训练完了，该globle迭代了，这个时候mask后上传
            self.mask()
            self.push_pull()#传送

    def train(self):#data哪来的？？
        loss_avg = 0
        cnt = 0
        for x, y in self.data:
            cnt += 1
            self.optimizer.zero_grad()
            x = x.to(self.device)
            y = y.to(self.device)
            output = self.model(x)
            loss = self.criterion(output, y)
            loss.backward()
            loss_avg += loss.item()
            self.optimizer.step()
        return loss_avg / cnt

    def mask(self):
        #建图
        self.neighbors_list = param.findNeighbors(param.root_seed, self.global_epoch, self.n_clients, self.idx,
                                                  self.neighborhood_size)

        # Download public keys of neighbors from PKI file
        # NOTE: the ABidxES framework has client idx starting from 1.
        neighbor_pubkeys = {}
        for idx in self.neighbors_list:
            try:
                hdr = 'pki_files/client' + str(idx) + '.pem'
                f = open(hdr, "rt")
                key = ECC.import_key(f.read())
                f.close()
            except IOError:
                raise RuntimeError("No such file. Run setup_pki.py first.")
            pk = key.pointQ
            neighbor_pubkeys[idx] = pk

            # compute pairwise masks r_ij
        neighbor_pairwise_secret_group = {}  # g^{a_i a_j} = r_ij in group
        neighbor_pairwise_secret_bytes = {}

        for idx in self.neighbors_list:
            neighbor_pairwise_secret_group[idx] = self.secret_key * neighbor_pubkeys[idx]
            # hash the g^{ai aj} to 256 bits (16 bytes)
            px = (int(neighbor_pairwise_secret_group[idx].x)).to_bytes(self.key_length, 'big')
            py = (int(neighbor_pairwise_secret_group[idx].y)).to_bytes(self.key_length, 'big')

            hash_object = SHA256.new(data=(px + py))
            neighbor_pairwise_secret_bytes[idx] = hash_object.digest()[0:self.key_length]
        ###neighbor_pairwise_secret_bytes是256bit的g^{ai aj}
        neighbor_pairwise_mask_seed_group = {}
        neighbor_pairwise_mask_seed_bytes = {}

        """Mapping group elements to bytes.
            compute h_{i, j, t} to be PRF(r_ij, t)
            map h (a binary string) to a EC group element
            encrypt the group element
            map the group element to binary string (hash the x, y coordinate)
        """
        for idx in self.neighbors_list:
            round_number_bytes = self.global_epoch.to_bytes(16, 'big')
            ###怎么都是byet形式
            h_ijt = ChaCha20.new(key=neighbor_pairwise_secret_bytes[idx], nonce=param.nonce).encrypt(
                round_number_bytes)
            h_ijt = str(int.from_bytes(h_ijt[0:4], 'big') & 0xFFFF)

            # map h_ijt to a group element
            dst = ecchash.test_dst("P256_XMD:SHA-256_SSWU_RO_")
            neighbor_pairwise_mask_seed_group[idx] = ecchash.hash_str_to_curve(msg=h_ijt, count=2,
                                                                              modulus=self.prime, degree=ecchash.m,
                                                                              blen=ecchash.L,
                                                                              expander=ecchash.XMDExpander(dst,
                                                                                                           hashlib.sha256,
                                                                                                           ecchash.k))

            px = (int(neighbor_pairwise_mask_seed_group[idx].x)).to_bytes(self.key_length, 'big')
            py = (int(neighbor_pairwise_mask_seed_group[idx].y)).to_bytes(self.key_length, 'big')

            hash_object = SHA256.new(data=(px + py))  # 椭圆曲线点上坐标之和
            neighbor_pairwise_mask_seed_bytes[idx] = hash_object.digest()[0:self.key_length]  # hij椭圆曲线点上坐标之和

        state_dict = self.model.state_dict()

        # for key, value in state_dict.items():
        #     print(f"Parameter: {key}")
        #     print(f"Shape: {value.shape}")
        #     print(value)
        #     break
        prg_pairwise = {}
        for idx in self.neighbors_list:
            prg_pairwise_holder = ChaCha20.new(key=neighbor_pairwise_mask_seed_bytes[idx],
                                               nonce=param.nonce)  ##在这里联合M
            # 生成足够长的伪随机数据
            #data_length = sum(key.numel() for key in state_dict.values())
            data = b"secr" * 65536 # 确保长度足够
            #data = data[:data_length]  # 修正到正确的长度
            prg_pairwise[idx] = prg_pairwise_holder.encrypt(data)


        # 将所有参数展平并合并到一个向量中
        float_vec = np.empty(0)#一个空的 float_vec 数组
        for key in state_dict.keys():
            par = state_dict[key]
            #print(f"{key}: {state_dict[key]}")
            float_vec = np.concatenate((float_vec, np.array(state_dict[key]).flatten()))
        padding = 65536 - sum(key.numel() for key in state_dict.values())
        float_vec = np.concatenate((float_vec, np.zeros(padding)))

        vec = np.vectorize(lambda d: d * pow(2, 12))(float_vec).astype(self.vector_dtype)
        #flattened_params_array = np.array(flattened_params)
        #vec = np.vectorize(lambda d: d * pow(2, 12))(flattened_params).astype(self.vector_dtype)
        #vec = (flattened_params_array * pow(2, 12)).astype(self.vector_dtype)
        vec_prg_pairwise = {}
        for idx in self.neighbors_list:
            print(self.idx, idx)
            vec_prg_pairwise[idx] = np.frombuffer(prg_pairwise[idx], dtype=self.vector_dtype)
            # print(f"Shape: {vec_prg_pairwise[idx].shape}")
            # print(vec_prg_pairwise[idx])#就一个vec了？？？那每个key的值加一样的得了 和每个用户是一个超长向量

        # state_dict = self.model.state_dict()
        # for key in state_dict.keys():
        #     state_dict[key] += torch.tensor(vec_prg_pairwise[key].reshape(state_dict[key].shape),
        #                                     dtype=state_dict[key].dtype)
        #     print(vec_prg_pairwise[idx])

            # if len(vec_prg_pairwise[id]) != self.vector_len:#所以data是
            #     raise RuntimeError("vector length error")

            assert len(vec) == len(vec_prg_pairwise[idx])
            if self.idx - 1 < idx:
                # state_dict = self.model.state_dict()
                # avg_mask = vec_prg_pairwise[id]
                vec +=vec_prg_pairwise[idx]
                    #avg_mask = avg_mask[param_len:]  # Move to the next portion of the mask

            #elif self.idx - 1 > idx:
                # state_dict = self.model.state_dict()
                # avg_mask = vec_prg_pairwise[id]
               #vec -= vec_prg_pairwise[idx]

            # 将展平后的参数重新分配回原始的形状
            # start_idx = 0
            # for key in state_dict.keys():
            #     param_shape = state_dict[key].shape
            #     param_len = state_dict[key].numel()
            #     reshaped_param = torch.tensor(flattened_params[start_idx:start_idx + param_len].reshape(param_shape),
            #                                   dtype=state_dict[key].dtype)
            #     state_dict[key] = reshaped_param
            #     start_idx += param_len

            # 打印结果以验证
            # for key in state_dict.keys():
            #     print(f"{key}: {state_dict[key]}")
            # else:
            #     raise RuntimeError("self id - 1 =", self.id - 1, " should not appear in neighbor_list",
            #                        self.neighbors_list)

        # reshaped_mask = torch.tensor(avg_mask[:param_len].reshape(param_shape), dtype=torch.int32)
        # state_dict['conv1.weight'] += reshaped_mask
        # avg_mask = avg_mask[param_len:]

        # state_dict = self.model.state_dict()
        # for key in state_dict.keys():
        #     avg_mask = vec_prg_pairwise[id]
        #     param_shape = state_dict[key].shape#(4, 1, 3, 3)  # 模型参数的形状
        #     param_len = state_dict[key].numel()#4*1*3*3 = 36 参数的总长度
        #     reshaped_mask = torch.tensor(avg_mask[:param_len].reshape(param_shape), dtype=state_dict[key].dtype)
        #     state_dict[key] += reshaped_mask
        #     avg_mask = avg_mask[param_len:]  # Move to the next portion of the mask

    def push_pull(self):
        client_socket = socket.socket()
        client_socket.bind((self.ip, self.port))
        client_socket.connect((self.server_ip, self.server_port))
        #pickle.loads 将接收到的数据反序列化为模型参数pickle.dumps(float_vec)
        #client_socket.sendall(pickle.dumps(float_vec))
        # client_socket.sendall(pickle.dumps([self.sample_num, self.model.state_dict()]))
        client_socket.sendall(b'stop!')

        self.model.load_state_dict(self.client_recv(client_socket))
        client_socket.setsockopt(socket.SOL_SOCKET, socket.SO_LINGER, struct.pack('ii', 1, 0))
        client_socket.close()

    @staticmethod
    def client_recv(client_socket):
        new_para = b''
        tmp = client_socket.recv(1024)
        while tmp:
            if tmp.endswith(b'stop!'):
                new_para += tmp[:-5]
                break
            new_para += tmp
            tmp = client_socket.recv(1024)
        return pickle.loads(new_para)




