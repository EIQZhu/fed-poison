from Cryptodome.Random import get_random_bytes
from Cryptodome.Cipher import AES, ChaCha20
import numpy as np
#import pandas as pd
import math
#随机数v 不可预测的随机数

vector_len = 16000
vector_type = 'uint32'
root_seed = get_random_bytes(32)
nonce = b'\x00\x00\x00\x00\x00\x00\x00\x00'

def parse_segment_to_list(segment, num_choose, bits_per_client, bytes_per_client):
    cur_ls = set()
    # take a segment (byte string), parse it to a list
    for i in range(num_choose):
        cur_bytes = segment[i * bytes_per_client: i *
                                                  bytes_per_client + bytes_per_client]

        cur_no = int.from_bytes(cur_bytes, 'big') & (
                (1 << bits_per_client) - 1)

        cur_ls.add(cur_no)

    return cur_ls
# choose neighbors
def findNeighbors(root_seed, current_iteration, num_clients, idx, neighborhood_size):
    neighbors_list = set()  # a set, 无序，且不能重复 instead of a list

    # compute PRF(root, iter_num), output a seed. can use AES
    prf = ChaCha20.new(key=root_seed, nonce=nonce)
    current_seed = prf.encrypt(current_iteration.to_bytes(32, 'big'))

    # compute PRG(seed), a binary string
    prg = ChaCha20.new(key=current_seed, nonce=nonce)

    # compute number of bytes we need for a graph
    num_choose = math.ceil(math.log2(num_clients))  # number of neighbors I choose
    num_choose = num_choose * neighborhood_size

    bytes_per_client = math.ceil(math.log2(num_clients) / 8)
    segment_len = num_choose * bytes_per_client
    num_rand_bytes = segment_len * num_clients
    data = b"a" * num_rand_bytes
    graph_string = prg.encrypt(data)

    # find the segment for myself
    my_segment = graph_string[(idx - 1) *
                              segment_len: (idx - 1) * (segment_len) + segment_len]

    # define the number of bits within bytes_per_client that can be convert to int (neighbor's idx)
    bits_per_client = math.ceil(math.log2(num_clients))
    # default number of clients is power of two
    for i in range(num_choose):
        tmp = my_segment[i * bytes_per_client: i *
                                               bytes_per_client + bytes_per_client]
        tmp_neighbor = int.from_bytes(
            tmp, 'big') & ((1 << bits_per_client) - 1)

        if tmp_neighbor == idx - 1:
            # print("client", self.idx, " random neighbor choice happened to be itself, skip")
            continue
        if tmp_neighbor in neighbors_list:
            # print("client", self.idx, "already chose", tmp_neighbor, "skip")
            continue
        neighbors_list.add(tmp_neighbor)

    # now we have a list for who I chose
    # find my idx in the rest, see which segment I am in. add to neighbors_list
    for i in range(num_clients):
        if i == idx - 1:
            continue
        seg = graph_string[i * segment_len: i *
                                            (segment_len) + segment_len]
        ls = parse_segment_to_list(
            seg, num_choose, bits_per_client, bytes_per_client)
        if idx - 1 in ls:
            # add current segment owner into neighbors_list
            neighbors_list.add(i)

    return neighbors_list
