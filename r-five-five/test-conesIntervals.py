###################################################################################################################
# Enumerating graphs in R(K4,J5,n) which 13 < n < 19
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

# Example R(4,4,14)
with open('r44_14.g6', 'r') as file:
    original = file.read().splitlines()
r44_14 = [decodeG6(graph) for graph in original[:100]]


# Find feasible cones or subsets of the original graph that do not contain triangles.
# Helper Function: Finds indices of adjacent vertices
def neighboring(vertex, graph):
  row = np.array(graph[vertex])
  neighbors = np.where(row == 1)[0]
  return neighbors

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
    edgeVariables = dict()
    # Add triangle conditions
    for i in vertices:
        neighbors = neighboring(i, H) # Neighors connected to vertex
        newNeighbors = np.setdiff1d(neighbors, np.arange(i))
        for j in range(len(newNeighbors)):
            for k in range(len(newNeighbors)-j-1):
                if H[newNeighbors[j]][newNeighbors[j+k+1]] == 1: # Edge connected to vertex
                    model.AddBoolOr([variables[i].Not(), variables[newNeighbors[j]].Not(), variables[newNeighbors[j+k+1]].Not()])
    # Find feasible cones
    solver = cp_model.CpSolver()
    solution_printer = VarArraySolutionPrinter(variables)
    solver.parameters.enumerate_all_solutions = True
    status = solver.Solve(model, solution_printer)
    bitSolutions = solution_printer.solution_vector()
    feasibleCones = [{vertices[i] for i in range(len(vertices)) if solution[i] == 1} for solution in bitSolutions]
    return feasibleCones

def maximalFeasibleCones(H):
    vertices = range(len(H))
    model = cp_model.CpModel()
    variables = [model.NewBoolVar('v'+str(i)) for i in range(len(H))]
    edgeVariables = dict()
    # Add triangle conditions
    triangles = []
    for i in vertices:
        neighbors = neighboring(i, H) # Neighors connected to vertex
        newNeighbors = np.setdiff1d(neighbors, np.arange(i))
        for j in range(len(newNeighbors)):
            for k in range(len(newNeighbors)-j-1):
                if H[newNeighbors[j]][newNeighbors[j+k+1]] == 1: # Edge connected to vertex
                    model.AddBoolOr([variables[i].Not(), variables[newNeighbors[j]].Not(), variables[newNeighbors[j+k+1]].Not()])
                    triangles.append([i, newNeighbors[j], newNeighbors[j+k+1]])
                    # We need to add intermediate edge variables in order to define maximility
                    if (str(newNeighbors[j])+'e'+str(newNeighbors[j+k+1])) not in edgeVariables:
                        edgeVariables[str(newNeighbors[j])+'e'+str(newNeighbors[j+k+1])] = model.NewBoolVar(str(newNeighbors[j])+'e'+str(newNeighbors[j+k+1]))
                        model.AddAllowedAssignments([variables[newNeighbors[j]], variables[newNeighbors[j+k+1]], edgeVariables[str(newNeighbors[j])+'e'+str(newNeighbors[j+k+1])]],
                            [(True, True, True), (True, False, False), (False, True, False), (False, False, False)])
                    if (str(i)+'e'+str(newNeighbors[j])) not in edgeVariables:
                        edgeVariables[str(i)+'e'+str(newNeighbors[j])] = model.NewBoolVar(str(i)+'e'+str(newNeighbors[j]))
                        model.AddAllowedAssignments([variables[i], variables[newNeighbors[j]], edgeVariables[str(i)+'e'+str(newNeighbors[j])]],
                            [(True, True, True), (True, False, False), (False, True, False), (False, False, False)])
                    if (str(i)+'e'+str(newNeighbors[j+k+1])) not in edgeVariables:
                        edgeVariables[str(i)+'e'+str(newNeighbors[j+k+1])] = model.NewBoolVar(str(i)+'e'+str(newNeighbors[j+k+1]))
                        model.AddAllowedAssignments([variables[i], variables[newNeighbors[j+k+1]], edgeVariables[str(i)+'e'+str(newNeighbors[j+k+1])]],
                            [(True, True, True), (True, False, False), (False, True, False), (False, False, False)])

        # Add maximal conditions
        orClause = [variables[i]]
        for triangle in triangles:
            if i in triangle:
                badEdge = [vertex for vertex in triangle if vertex != i]
                orClause.append(edgeVariables[str(badEdge[0])+'e'+str(badEdge[1])])
        model.AddBoolOr(orClause)

    # Find maximalfeasible cones
    solver = cp_model.CpSolver()
    solution_printer = VarArraySolutionPrinter(variables)
    solver.parameters.enumerate_all_solutions = True
    status = solver.Solve(model, solution_printer)
    bitSolutions = solution_printer.solution_vector()
    maxCones = [{vertices[i] for i in range(len(vertices)) if solution[i] == 1} for solution in bitSolutions]
    return maxCones

# Before putting our feasible cones into intervals, we can sort them in two ways.

# We can order them from biggest to smallest:
def bigToSmall(H):
    cones = feasibleCones(H)
    cones.sort(key=len, reverse=True)
    return cones

# We can order them with the maximal feasible cones first:
def maxFirst(H):
    cones = []
    maxCones = maximalFeasibleCones(H)
    maxCones.sort(key=len, reverse=True)
    for maxCone in maxCones:
        subCones = powerset(maxCone)
        subCones.remove(maxCone)
        for cone in subCones:
            if cone not in cones:
                cones.append(cone)
    cones.sort(key=len, reverse=True)
    return maxCones + cones

# Finds intervals without SAT
def intervals(H, feasibleCones):
    feasibleCones = copy.deepcopy(feasibleCones)
    vertices = {i for i in range(len(H))}
    intervals = []

    # We place feasible cones into intervals until there are no more feasible cones
    while feasibleCones != list():
        top = feasibleCones[0]
        bottom = set()

        for interval in intervals:
            # We only adjust the bottom if the intervals are not already distinct
            if not top.issuperset(interval[0]) or not bottom.issubset(interval[1]):
                continue

            # Otherwise we take away from the top vertices which in the bottom
            for vertex in vertices.difference(interval[1]):
                if vertex in top:
                    bottom.add(vertex)
                    break

        # We add another interval
        intervals.append([bottom, top])

        # We update our feasible cones
        full = powerset(top)
        for feasibleCone in full:
            if feasibleCone.issuperset(bottom):
                feasibleCones.remove(feasibleCone)

    return intervals

# Finds intervals with SAT
def intervalsSAT(H, feasibleCones):
    vertices = {i for i in range(len(H))}
    intervals = []
    # We place feasible cones into intervals until there are no more feasible cones
    while feasibleCones != list():
        top = feasibleCones[0]
        bottom = set()

        model = cp_model.CpModel()
        variables = [model.NewBoolVar('v'+str(i)) for i in range(len(H))]

        # We only adjust the bottom if the intervals are not already distinct
        for interval in intervals:
            if not top.issuperset(interval[0]):
                continue

            # We force the interval to be disjoint from the other intervals by adding vertices to the bottom
            model.AddBoolOr([variables[vertex] for vertex in vertices.difference(interval[1]) if vertex in top])

        # We want the smallest bottom
        model.Minimize(sum(variables))

        # We solve for the smallest bottom
        solver = cp_model.CpSolver()
        solution_printer = VarArraySolutionPrinter(variables)
        status = solver.Solve(model, solution_printer)
        bitSolution = solution_printer.solution_vector()[0]
        bottom = {i for i in range(len(H)) if bitSolution[i] == 1}
        # We add another interval
        intervals.append([bottom, top])

        # We update our feasible cones
        full = powerset(top)
        for feasibleCone in full:
            if feasibleCone.issuperset(bottom):
                feasibleCones.remove(feasibleCone)

    return intervals

# The best combination is maxFirst and intervalsSAT


# we first brute force test
H = r44_14[0]
# call cones result
def brute_extend(G, verts, extended):
    ''' Brute forces every combination of 3 vertices to find a triangle. 
    Returns True if vertex can be extended without creating triangles'''
    extent = verts # add the extended vertex
    extent.add(extended)
    for vs in combinations(extent, 3):
        i = vs[0]
        j = vs[1]
        k = vs[2]
        if (G[i][j] == 1 and G[j][k] == 1 and G[i][k] == 1):
            return False
    return True



def brute_testing(H):
    '''brute forces all possible cones, comparing the number found to Elisha's algorithm. 
    Expects Elisha's number to be exactly 1 more than brute force (accounting for emptyset). 
    Also tests the maximality of the maximal cones generated'''
    x1 = len(feasibleCones(H))
    # now we brute force, go through every subset up to size 8, .is_triangle_free()
    total_found = 0
    those_found = []
    G = H
    for b in range(8):
        # for subgraphs of G size 1 up to size 8
        for subg in combinations([x for x in range(len(H))], b+1):
            # for every set of 3 vertices in this subgraph, we see if they share an edge
            t_found = False
            for vs in combinations(subg, 3):
                i = vs[0]
                j = vs[1]
                k = vs[2]
                # check if K3
                if (G[i][j] == 1 and G[i][k]==1 and G[j][k]==1):
                    t_found = True #TRIANGLE FOUND!!!
                    break
            if not t_found:
                total_found +=1
                those_found.append(subg)
    print("feasibleCones is correct: ", x1-total_found == 1) # should be true
    mcones = maximalFeasibleCones(H)
    messedUp = False
    # to check a maximal feasible cone, check that adding any vertex results in a triangle, 
    for cone in mcones:
        extendies = {x for x in range(len(H))} - cone # vertices in H setminus cone
        
        for v in extendies:
            if brute_extend(H,cone,v) == True:
                print("NOOOOO NOt maximal: ", cone)
                messedUp = True
            else:
                #print("Maximality verified: ", cone)
                pass
    if not messedUp:
            print("maximalFeasibleCones is correct ")   
    else:
        print("maximalFeasibleCones messed up") 

for i in range(len(r44_14)):
    brute_testing(r44_14[i])
