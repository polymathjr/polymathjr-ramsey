###################################################################################################################
# Encoder and decoder in Python for Graph6 format
###################################################################################################################

# Function to reformat graph6 formatted graphs into adjacency matrices
def decodeG6(graph):
    adjacencyMatrix = []
    # Gets dimensions of matrix as well as its corresponding bit vector
    dimension = 0
    bitVect = ""
    if ord(graph[0]) != 126 and ord(graph[1]) != 126:
        dimension += ord(graph[0]) - 63
        for character in graph[1:]:
            bitVect += bin(ord(character) - 63)[2:].zfill(6)
    elif ord(graph[1]) != 126:
        for character in range(1,4):
            dimension += (ord(graph[character]) - 63) * 2**(18 - 6*i)
        for character in graph[4:]:
            bitVect += bin(ord(character) - 63)[2:].zfill(6)
    elif ord(graph[0]) == 126:
        for character in (2, 8):
            dimension += (ord(graph[character]) - 63) * 2**(18 - 6*i)
        for character in graph[8:]:
            bitVect += bin(ord(character) - 63)[2:].zfill(6)
    bitVect = bitVect[:int((dimension - 1) * dimension / 2)]
    # Constructs adjacency matrix using its dimensions and bit vector
    adjacencyMatrix = [[0 for i in range(dimension)] for j in range(dimension)]
    pointer = 0
    for column in range(1, dimension):
        for row in range(column):
            adjacencyMatrix[row][column] = int(bitVect[pointer])
            adjacencyMatrix[column][row] = int(bitVect[pointer])
            pointer += 1
    return adjacencyMatrix

# Note: I also tested it compared with Sage's operation, and after converting data types, they proved equivalent
# To test for yourself, make sure to have downloaded the appropiate text files
"""
# Testing it on r39_35
with open('r39_35.g6', 'r') as file:
    original = file.read().splitlines()
r39_35 = decodeG6(original[0])
print(r39_35)


# Testing it on r35_10
with open('r35_10.g6', 'r') as file:
    original = file.read().splitlines()
r35_10 = [decodeG6(graph) for graph in original]
print(r35_10)
"""

# Helper Function #1
def R(x):
    x += '0' * (6 - (len(x) % 6))
    x = [int(x[6*i : 6*i + 6], 2) + 63 for i in range(int(len(x)/6))]
    return x

# Helper Function #2
def N(n):
    if 0 <= n and n <= 62:
        return [n + 63]
    if 63 <= n and n <= 258047:
        return [126] + R(bin(n)[2:].zfill(18))
    if 259048 <= n and n <= 68719476735:
        return [126, 126] + R(bin(n)[2:].zfill(36))

# Function to convert from a graph's adjacency matrix to its compressed form
def compressG6(adjacencyMatrix):
    bitVect = ""
    n = len(adjacencyMatrix)
    head = N(n)
    columnAdjacencyMatrix = [[adjacencyMatrix[i][j] for i in range(n)] for j in range(n)]
    for i in range(1, n):
        for j in range(i):
            bitVect += str(adjacencyMatrix[j][i])
    tail = R(bitVect)
    code = head + tail
    return "".join(chr(num) for num in code)

"""
# Testing it on r39_35
with open('r39_35.g6', 'r') as file:
    original = file.read().splitlines()
r39_35 = decodeG6(original[0])
new = [compressG6(r39_35)]
print(new)
print(original)
print(original == new)

# Testing it on r35_10
with open('r35_10.g6', 'r') as file:
    original = file.read().splitlines()
r35_10 = [decodeG6(graph) for graph in original]
new = [compressG6(r35_10[i]) for i in range(len(r35_10))]
print(new)
print(original)
print(original == new)
"""
