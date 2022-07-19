from sage.all_cmdline import *   # import sage library

def beta(u, v):
    '''Takes in two 2n dimensional vectors over F_2, and outputs a 0 or 1.'''
    
    if len(u) != len(v) or len(u) % 2 != 0: 
        raise Exception('Input vectors need to have equal and even dimension.')
    
    B = 0
    for i in range(0, len(u), 2): 
        B += u[i] * v[i + 1] - u[i + 1] * v[i]
    
    return B



def conlon_ferber(n):
    '''Returns the Conlon-Ferber graph on F_2^n - {0}'''
    
    if n % 2 != 0:
        raise Exception('Conlon-Ferber graph must be defined on an even dimension.')

    V = list(VectorSpace(GF(2), n))
    V.remove(vector(GF(2), [0] * n))
    for vec in V: vec.set_immutable()

    return Graph([V, lambda u, v: beta(u, v) == 1])