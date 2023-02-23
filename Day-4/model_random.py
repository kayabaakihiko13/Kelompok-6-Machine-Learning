
import time
import numpy as np
import os
"""

===========================
        make own random
===========================

"""
BPF = 53  
def random():
    """
    random ini berfungsi untuk menampilkan angka dengan sesuai pseucode 
    ini 
    >>> random()
    0.153196923667565
    >>> random()
    0.734023273922503
    """
    t=int(time.time()*1000)
    x=t%(2**32)
    while True:

        x=(1664525*x+12345)%(2**32)
        return x/(2**32)


def getrandbits(k):
    """
    konsep dari getrandbits ini menggunakan bits acak yang relative
    dengan melewati dua waktu
    refence:
    https://en.wikipedia.org/wiki/Linear_congruential_generator

    """
    bytes_need=(k+7)
    randbytes=os.urandom(bytes_need)
    randbit=int.from_bytes(randbytes,byteorder='big')
    return randbit>>(bytes_need*8-k)


# make important 
def randbelow(n):
    """
    
    """
    if n<=0:
        raise ValueError("the input cant be less from zero")
    k=n.bit_length()        
    return getrandbits(k) % n


def randomint(a,b,size=1):
    result=[]
    for _ in range(size):
        result.append(randbelow(b-a+1))
    return np.array(result)


def randomrange(a,b,size):
    result=[]
    for _ in range(size):
        result.append(randbelow(b-a))
    return np.array(result)


def random_uniform(size=1):
    if size<=0:
        raise ValueError("Can't put size value lower than 1")
    
    memo=[]
    for _ in range(size): 
        memo.append(0 + (1 - 0) * random())
    return np.array(memo)

## Tahap pembagunan
# def random_uniform(low,high,size=1):
#     result=np.array([])
#     for _ in range(size):
#         result=np.append(result,low+(high-low)*randomint(low,high,size=1))
#     return result

if __name__=="___main__":
    import doctest

    doctest.testmod(verbose=True)