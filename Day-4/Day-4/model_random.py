import time
import os
import numpy as np
"""

===========================
        make own random
===========================

"""


def random(size=1):
    """
    angka acak ini menggunakan basis LCG Linear congruential generator
    refence:

    https://en.wikipedia.org/wiki/Linear_congruential_generator

    >>> random()
    0.153196923667565
    >>> random()
    0.734023273922503
    """
    t = int(time.time() * 1000)
    x = t % (2 ** 31)
    memo=[]
    for _ in range(size):
        x = (1664525 * x + 12345) % (2 ** 31)
        hitung = x / (2 ** 31)
        memo.append(hitung)
    return np.array(memo)

def seed(number):
    def generator(size=None):
        t=seed
        x = t % (2 ** 31)
        memo=[]
        while size is None | len(memo)<size:
            x = (1664525 * x + 12345) % (2 ** 31)
            hitung=x/(2*31)
            memo.append(hitung)
        if size is None:
            memo[0]
        return memo
    return random


def getrandbits(k):
    """
    konsep dari getrandbits ini menggunakan bits acak yang relative
    dengan melewati dua waktu atau dengan menggunakan generator byte

    """
    bytes_need = (k + 7) // 8
    randbytes = os.urandom(bytes_need)
    randbit = int.from_bytes(randbytes, byteorder='big')
    return randbit >> (bytes_need*8-k)


def generator_bit(n):
    if n <= 0:
        raise ValueError("the input cant be less from zero")
    k = n.bit_length()
    return getrandbits(k) % n


def randomint(a, b, size=1):
    result = []
    for _ in range(size):
        result.append(generator_bit(b - a + 1) + a)
    return np.array(result).reshape(-1,1)


def randomrange(a, b, size=1):
    result = []
    for _ in range(size):
        result.append(generator_bit(b - a) + a)
    return np.array(result).shape(-1,1)

def randuni(low=0,high=1,size=1):
    memo=np.array([])
    for _ in range(size):
        hitung=low+(high-low)*random()
        memo=np.append(memo,hitung)
    return memo.shape(-1,1)

def generate_dummy_data_single(n_samples, beta_0, beta_1, noise_scale):
    x = np.linspace(0, 1, n_samples)
    noise = randuni(size=n_samples, low=-noise_scale, high=noise_scale)
    y = beta_0 + beta_1 * x + noise
    return x, y


"""
import random

def generate_random_float(lower_bound, upper_bound):
    return random.random() * (upper_bound - lower_bound) + lower_bound

random_float = generate_random_float(0.0, 1.0)
print(random_float)

"""


if __name__ == "__main__":
    import doctest
    doctest.testmod(verbose=True)
