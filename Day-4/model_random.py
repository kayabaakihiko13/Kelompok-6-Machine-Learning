import struct
import time
import numpy as np
import os
"""

===========================
        make own random
===========================

"""

def random():
    """
    random ini berfungsi untuk menampilkan angka dengan sesuai pseucode 
    ini 
    """
    seed=0
    multiplier=1664525
    increment=1013904223
    modulus=pow(2,31)
    seed=(multiplier*seed+increment)%modulus
    return seed/modulus

def getrandbits(k):
    bytes_need=(k+7)
    randbytes=os.urandom(bytes_need)
    randbit=int.from_bytes(randbytes,byteorder='big')
    return randbit>>(bytes_need*8-k)


# make important 
def randbelow(n):
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


def random_interval():
    current_time=time.time()
    seed=randbelow((current_time)-int(current_time))
    return seed

## Tahap pembagunan
# def random_uniform(low,high,size=1):
#     result=np.array([])
#     for _ in range(size):
#         result=np.append(result,low+(high-low)*randomint(low,high,size=1))
#     return result

if __name__=="___main__":
    import doctest

    doctest.testmod(verbose=True)