import numpy as np
import random

def dot_product(x, y):
    if len(x) != len(y):
        raise Exception("Error: Length of vectors do not agree")
    if x[0] > 1:
        raise Exception("Are you working with binary vectors?")
    sum = 0

    return x.y

def add(x, y):
    if len(x) != len(y):
        raise Exception("Length of vectors do not agree")

    z = []
    for i in range(len(x)):
        z.append(x[i] + y[i])
    print("Checkpoint 1: The sum of these two vectors has been computed.")
    return z
