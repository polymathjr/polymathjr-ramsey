###################################################################################################################
# Enumerating graphs in R(K4,J5,n) which 13 < n < 19
# It is purposely a little inefficient so it will be more compatible with gluing H to G instead of the other way around.
###################################################################################################################
import numpy as np
from ortools.sat.python import cp_model
from itertools import chain, combinations
import copy


# Useful function for later on (I just copied this from the internet ;] )
def powerset(A):
    length = len(A)
    return [{e for e, b in zip(A, f'{i:{length}b}') if b == '1'} for i in range(2 ** length)]

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

# Find feasible cones or subsets of the original graph that do not contain triangles.
# Helper Function: Finds indices of adjacent vertices
def neighboring(vertex, graph):
  row = np.array(graph[vertex])
  neighbors = np.where(row == 1)[0]
  return neighbors

def notNeighboring(vertex, graph):
  row = np.array(graph[vertex])
  notNeighbors = np.where(row == 0)[0]
  return notNeighbors

# Needed to find solutions to SAT problem
class VarArraySolutionPrinter(cp_model.CpSolverSolutionCallback):
    # Print intermediate solutions.

    def __init__(self, variables):
        cp_model.CpSolverSolutionCallback.__init__(self)
        self.__variables = variables
        self.__solution_count = 0
        self.__solution_vector = []
    def on_solution_callback(self):
        self.__solution_count += 1
        self.__solution_vector.append([self.Value(v) for v in self.__variables])
    def solution_count(self):
        return self.__solution_count
    def solution_vector(self): # Solution vector, tells us the values of the variables
        return self.__solution_vector

def feasibleCones(H):
    vertices = range(len(H))
    model = cp_model.CpModel()
    variables = [model.NewBoolVar('v'+str(i)) for i in range(len(H))]

    # Add triangle conditions
    for i in vertices:
        neighbors = neighboring(i, H) # Neighors connected to vertex
        newNeighbors = np.setdiff1d(neighbors, np.arange(i))
        for j in range(len(newNeighbors)):
            for k in range(len(newNeighbors)-j-1):
                if H[newNeighbors[j]][newNeighbors[j+k+1]] == 1: # Edge connected to vertex
                    model.AddBoolOr([variables[i].Not(), variables[newNeighbors[j]].Not(), variables[newNeighbors[j+k+1]].Not()])

    # Add complement triangle conditions
    for i in vertices:
        notNeighbors = notNeighboring(i, H) # Not neighors connected to vertex
        newNotNeighbors = np.setdiff1d(notNeighbors, np.arange(i+1))
        for j in range(len(newNotNeighbors)):
            for k in range(len(newNotNeighbors)-j-1):
                if H[newNotNeighbors[j]][newNotNeighbors[j+k+1]] == 0: # Not edge not connected to vertex
                    model.AddBoolOr([variables[i], variables[newNotNeighbors[j]], variables[newNotNeighbors[j+k+1]]])

    solver = cp_model.CpSolver()
    solution_printer = VarArraySolutionPrinter(variables)
    solver.parameters.enumerate_all_solutions = True
    status = solver.Solve(model, solution_printer)
    bitSolutions = solution_printer.solution_vector()
    feasibleCones = [{vertices[i] for i in vertices if solution[i] == 1} for solution in bitSolutions]
    return feasibleCones

# We now introduce some single cone collapsing rules!

# First we define our helper functions:
def h1(sub, graph): # returns vertices connected to a vertex in our subgraph
    tempSub = sub.copy()
    solution = set()
    for i in range(len(tempSub)):
        vertex = tempSub.pop()
        neighbors = set(neighboring(vertex, graph))
        solution.update(neighbors)
    return solution

def h2(sub, graph): # returns vertices that are not connected to a vertex not in our subgraph
    vertices = {i for i in range(len(graph))}
    compSub = vertices.difference(sub)
    solution = set()
    for i in range(len(compSub)):
        vertex = compSub.pop()
        notNeighbors = set(notNeighboring(vertex, graph))
        notNeighbors.discard(vertex)
        solution.update(notNeighbors)
    return solution

def h3(sub, graph): # returns vertices that are not connected to two unconnected vertices not in our subgraph
    vertices = {i for i in range(len(graph))}
    compSub = vertices.difference(sub)
    ordered = list(compSub)

    # Find unconnected edges outside our subgraph
    badEdges = []
    length = len(ordered)
    for i in range(length - 1):
        for j in range(i + 1, length):
            if graph[ordered[i]][ordered[j]] == 0:
                badEdges.append({ordered[i], ordered[j]})

    solution = set()
    for badEdge in badEdges:
        vertex1 = badEdge.pop()
        vertex1Enemies = set(notNeighboring(vertex1, graph))
        vertex1Enemies.remove(vertex1)
        vertex2 = badEdge.pop()
        vertex2Enemies = set(notNeighboring(vertex2, graph))
        vertex1Enemies.remove(vertex2)

        solution.update(vertex1Enemies.intersection(vertex2Enemies))
    return solution

def h4(sub, graph): # returns vertices that are not connected to an edge not in our subgraph or are connected by one edge to a blue edge not our subgraph
    vertices = {i for i in range(len(graph))}
    compSub = vertices.difference(sub)
    ordered = list(compSub)

    # Find blue and red edges outside our subgraph
    blueEdges = []
    redEdges = []
    length = len(ordered)
    for i in range(length - 1):
        for j in range(i + 1, length):
            if graph[ordered[i]][ordered[j]] == 0:
                blueEdges.append({ordered[i], ordered[j]})
            elif graph[ordered[i]][ordered[j]] == 1:
                redEdges.append({ordered[i], ordered[j]})

    solution = set()
    for redEdge in redEdges:
        vertex1 = redEdge.pop()
        vertex1Enemies = set(notNeighboring(vertex1, graph))
        vertex2 = redEdge.pop()
        vertex2Enemies = set(notNeighboring(vertex2, graph))
        solution.update(vertex1Enemies.intersection(vertex2Enemies))

    for blueEdge in blueEdges:
        vertex1 = blueEdge.pop()
        vertex1Enemies = set(neighboring(vertex1, graph))
        vertex2 = blueEdge.pop()
        vertex2Enemies = set(neighboring(vertex2, graph))
        solution.update(vertex1Enemies.symmetric_difference(vertex2Enemies))
    return solution

# Edge rule
def k2(c1, c2, H):
    common = c1.intersection(c2)
    shared = c1.union(c2)
    # Has to be connected by a blue edge to any edge in H
    if common.intersection(h1(common, H)) != set():
        return False
    # Has to be connected to any independent 3-set in H
    elif not h3(shared, H).issubset(shared):
        return False
    else:
        return True

# Blue edge rule
def e2(c1, c2, H):
    common = c1.intersection(c2)
    shared = c1.union(c2)
    # Has to be connected by two edges to any independent 3-set
    if not h3(shared, H).issubset(common):
        return False
    # Has to be connected to any independent J3
    elif not h4(shared, H).issubset(shared):
        return False
    else:
        return True

# Blue J3 rule
def j3(c1, c2, c3, H):
    shared = c1.union(c2,c3)
    # Has to be connected to every blue edge
    if not h2(shared, H).issubset(shared):
        return False
    else:
        return True

# Independent 3-set rule
def e3(c1, c2, c3, H):
    vertices = {i for i in range(len(H))}
    shared = c1.union(c2, c3)
    c12 = c1.intersection(c2)
    c13 = c1.intersection(c3)
    c23 = c2.intersection(c3)
    pairShared = c12.union(c13, c23)
    # Has to be connected by two edges to any blue edge
    if not h2(shared, H).issubset(pairShared):
        return False
    # Has to be connected to any edge
    elif not h1(vertices.difference(shared), H).issubset(shared):
        return False
    else:
        return True

# Blue J4 rule
def j4(c1, c2, c3, c4, H):
    vertices = set(range(len(H)))
    shared = c1.union(c2, c3, c4)
    # Has to be connect to every vertex
    if vertices != shared:
        return False
    else:
        return True

# Independent 4-set rule
def e4(c1, c2, c3, c4, H):
    vertices = set(range(len(H)))
    c12 = c1.intersection(c2)
    c13 = c1.intersection(c3)
    c14 = c1.intersection(c4)
    c23 = c2.intersection(c3)
    c24 = c2.intersection(c4)
    c34 = c3.intersection(c4)
    pairShared = c12.union(c13, c14, c23, c24, c34)
    # Has to be connected by two edges to every vertex
    if vertices != pairShared:
        return False
    else:
        return True

# Using these rules, we can make a general collapsing rule that takes as input the collapsed adjunct and parent and tests if they are compatible
def collapse(parentDict, adjunctDict, G, H):
    parentVertices = set(parentDict.keys())
    adjunctVertices = set(adjunctDict.keys())
    # First we check if the parent and the adjunct are compatible
    for vertex in parentVertices.intersection(adjunctVertices):
        if parentDict[vertex] != adjunctDict[vertex]:
            return False

    # We are only concerned with sub-structures involving the last vertex and the vertices between the adjunct vertex and the second to last vertex
    parentVertex = max(adjunctVertices)
    parentCone = adjunctDict[parentVertex]
    checkVertices = parentVertices.difference(adjunctVertices)

    # Find blue edges
    notNeighbors = set(notNeighboring(parentVertex, G)).intersection(checkVertices)
    for notNeighbor in notNeighbors:
        if not e2(parentCone, parentDict[notNeighbor], H):
            return False

    # Find edges:
    neighbors = set(neighboring(parentVertex, G)).intersection(checkVertices)
    for neighbor in neighbors:
        if not k2(parentCone, parentDict[neighbor], H):
            return False

    # Find independent 3-sets and first case of complement J3s:
    ind3set = []
    j3case1 = []
    unNeighbors = notNeighbors.copy()
    for i in range(len(notNeighbors) - 1):
        vertex = unNeighbors.pop()
        for unNeighbor in unNeighbors:
            if G[vertex][unNeighbor] == 0:
                ind3set.append({vertex, unNeighbor})
                if not e3(parentCone, parentDict[vertex], parentDict[unNeighbor], H):
                    return False

            elif G[vertex][unNeighbor] == 1:
                j3case1.append({vertex, unNeighbor})
                if not j3(parentCone, parentDict[vertex], parentDict[unNeighbor], H):
                    return False

    # Find second case of complement J3s:
    j3case2 = []
    for neighbor in neighbors:
        for notNeighbor in notNeighbors:
            if G[neighbor][notNeighbor] == 0:
                j3case2.append({neighbor, notNeighbor})
                if not j3(parentCone, parentDict[neighbor], parentDict[notNeighbor], H):
                    return False


    # Find independent 4-sets
    ind4set = []
    indLength = len(ind3set)
    for i in range(indLength - 2):
        for j in range(i + 1, indLength - 1):
            blue1 = ind3set[i]
            blue2 = ind3set[j]
            if blue1.intersection(blue2) != set():
                blue3 = blue1.symmetric_difference(blue2)
                if blue3 in ind3set[j + 1:]:
                    ind4 = blue1.union(blue2)
                    ind4set.append(ind4)

                    listInd4 = list(ind4)
                    if not e4(parentCone, listInd4[0], listInd4[1], listInd4[2], H):
                        return False

    j4case1 = []
    # Find first case of complement J4s
    for edge in j3case1:
        for notNeighbor in notNeighbors.difference(edge):
            listEdge = list(edge)
            vertex1 = listEdge[0]
            vertex2 = listEdge[1]
            if G[notNeighbor][vertex1] == 0 and G[notNeighbor][vertex2] == 0:
                j4case1.append({notNeighbor, vertex1, vertex2})
                if not j4(parentCone, parentDict[notNeighbor], parentDict[vertex1], parentDict[vertex2], H):
                    return False

    # Find the second case of complement J4s
    j4case2 = []
    for neighbor in neighbors:
        for ind3 in ind3set:
            listInd3 = list(ind3)
            vertex1 = listInd3[0]
            vertex2 = listInd3[1]
            if G[neighbor][vertex1] == 0 and G[neighbor][vertex2] == 0:
                j4case2.append({neighbor, vertex1, vertex2})
                if not j4(parentCone, parentDict[neighbor], parentDict[vertex1], parentDict[vertex2], H):
                    return False

    return True

# We now need to make a double tree with our (K3, J5)
class Node(object):
    adjunctSequence = [1, 1, 2, 2, 3, 3, 4] # This will be our sequence for determining adjuncts
    treeDict = {i:[] for i in range(1, 8)} # This will keep track of the each of the levels of the tree
    mainBranchesDict = {i:[] for i in range(1, 8)} # This will keep track of the main branches of the tree
    H = None

    def __init__(self, G, vertices, verticesDict, isMain, isCollapsed):
        self.vertices = vertices # This is a list of labelled vertices
        self.verticesDict = verticesDict # For each of these vertices, we need to keep track of which vertices in H they are connected to
        self.level = len(vertices) # How many vertices, determines the level on the tree of the node
        self.depth = len(verticesDict[vertices[0]]) # How many possible sequences of feasible cones are available at the node
        self.G = G
        self.isMain = isMain # Is it on the main branches of the tree?
        self.isCollapsed = isCollapsed # Has the algorithm already calculated the possible sequences of feasible cones at this node? (originally set to False)

        self.loc = len(Node.treeDict[self.level])
        Node.treeDict[self.level].append(self)
        if self.isMain:
            Node.mainBranchesDict[self.level].append(self)

        # We calculate the parent and adjunct, but only if the graph has more than two vertices
        self.parent = None
        self.adjunct = None
        if self.level > 1:
            # Sees if the parent already has been created
            parentVertices = self.vertices[:-1]
            parentGraph =  [[(self.G)[row][col] if (row in parentVertices and col in parentVertices) else None for col in range(len(self.G)) ] for row in range(len(self.G))]
            for node in Node.treeDict[self.level - 1]:
                if node.G == parentGraph:
                    self.parent = node
                    if self.isMain:
                        if node not in Node.mainBranchesDict[self.level - 1]:
                            node.isMain = True
                            Node.mainBranchesDict[self.level - 1].append(node)
                    break
            else:
                # Creates a parent node
                parentDict = {vertex:[] for vertex in parentVertices}
                newNode = Node(parentGraph, parentVertices, parentDict, self.isMain, False)
                self.parent = newNode

            # Finds or creates an adjunct node
            adjunctVertices = self.vertices[0 : Node.adjunctSequence[self.level - 1] - 1] + [self.vertices[-1]]
            adjunctGraph = [[self.G[row][col] if (row in adjunctVertices and col in adjunctVertices) else None for col in range(len(self.G)) ] for row in range(len(self.G))]
            for node in Node.treeDict[Node.adjunctSequence[self.level - 1]]:
                if node.G == adjunctGraph:
                    self.adjunct = node
                    break

            else:
                adjunctDict = {vertex: [] for vertex in adjunctVertices}
                newNode = Node(adjunctGraph, adjunctVertices, adjunctDict, False, False)
                self.adjunct = newNode

    # Gives a sequence of feasible cones at a certain depth in the dictionary
    def index(self, i):
        return {vertex:self.verticesDict[vertex][i] for vertex in self.verticesDict}

    # Collapses a node based on its ajunct and its parent
    def collapseNode(self):
        # Recursively, we need the parent and the adjunct collapsed
        if not (self.parent).isCollapsed:
            (self.parent).collapseNode()
        if not (self.adjunct).isCollapsed:
            (self.adjunct).collapseNode()

        for i in range((self.parent).depth):
            for j in range((self.adjunct).depth):
                if collapse((self.adjunct).index(j), (self.parent).index(i), self.G, Node.H):
                    newLayer = (self.parent).index(i)
                    newLayer[self.vertices[-1]] = ((self.adjunct).index(j))[self.vertices[-1]]
                    for vertex in self.verticesDict:
                        self.verticesDict[vertex].append(newLayer[vertex])
                    self.depth += 1
        self.isCollapsed = True
        return self

    def __str__(self):
        stringDict = ""
        for key in self.verticesDict:
            stringDict += str(key) + ": " + str(self.verticesDict[key]) + "\n"
        return "\n".join(str(row) for row in self.G) + "\n" + stringDict + "\n" + "\n".join(str(row) for row in Node.H) + "\n"


# R(K4,J6,10) to find feasible cones in
with open('k4k4e_10.g6', 'r') as file:
    original1 = file.read().splitlines()
k4j4 = [decodeG6(graph) for graph in original1]
# Make double tree with these R(K3, J5, 7)
with open('k3k5e_07.g6', 'r') as file:
    original2 = file.read().splitlines()
k3j5 = [decodeG6(graph) for graph in original2]

blankVertices = [i for i in range(len(k3j5[0]))]
blankDictionary = {vertex:[] for vertex in blankVertices}
for graph in k3j5:
    Node(graph, blankVertices, blankDictionary, True, False)

for H in k4j4[3:]:
    print("hello")
    cones = feasibleCones(H)
    length = len(cones)

    Node.H = H
    rootNodes = Node.treeDict[1]
    for root in rootNodes:
        root.depth = length
        root.verticesDict = {vertex:cones for vertex in root.vertices}
        root.isCollapsed = True

    for level in range(2, len(k3j5[0]) + 1):
        for node in Node.mainBranchesDict[level]:
            node.collapseNode()


    for node in Node.mainBranchesDict[len(k3j5[0])]:
        if node.verticesDict != blankDictionary:
            print(node)

    for level in Node.treeDict:
        for node in Node.treeDict[level]:
            node.depth = 0
            node.verticesDict = {vertex:[] for vertex in node.vertices}
            node.isCollapsed = False
