from sage.graphs.graph_input import from_graph6
from sage.all_cmdline import *

def isEdge(u, v):
    x = u[0] - v[0]; y = u[1] - v[1]; z = u[2] - v[2]
    return ((x != 0) and (x*z) == (y*y)) 
    
def kopparty(q):
    '''
    Returns Kopparty Graph defined over F_q
    '''
    try:
        k = GF(q, repr='int') # This gives us the elements of the finite fields as integers
    except:
        raise Exception("The order of a finite field must be a prime power")
    V = list(VectorSpace(k, 3)) # This gives us F^3_q
    for vec in V: 
        vec.set_immutable()
    return Graph([V, lambda u, v: isEdge(u, v)])

kp = kopparty(3)
kp.show()
print(kp.order(), kp.size(), kp.complement().size())



with open('r35_13.g6', 'r') as file: # Opens the file, and closes it once we exit this block. 
    graphs = file.readlines() 
print(graphs)
g = Graph() # Create an empty graph
from_graph6(g, graphs[0])
g.show()


kp9 = kopparty(9)
print("Graph constructed")
subg3 = kp9.subgraph_search(g, induced=True)
if subg3 is not None:
    print(subg3.vertices())

