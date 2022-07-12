import numpy as np
import random

def dot_product(x, y):
    if len(x) != len(y):
        raise Exception("Length of vectors do not agree").

    sum = 0
    for i in range(len(x)):
        sum += x[i] * y[i]
    return sum

def add(x, y): 
    return x + y
