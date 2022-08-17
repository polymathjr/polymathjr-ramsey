###################################################################################################################
# Enumerating graphs in R(K4,J5,n) which 13 < n < 19
###################################################################################################################
import numpy as np
from ortools.sat.python import cp_model

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
r44_14 = [decodeG6(graph) for graph in original[:50]]
#r44_14 = decodeG6(original[50])


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
    def on_solution_callback(self):
        self.__solution_count += 1

        for v in self.__variables:
            print('%s=%i' % (v, self.Value(v)), end=' ')
        print()
    def solution_count(self):
        return self.__solution_count

def maximalFeasibleCones(H):
    model = cp_model.CpModel()
    variables = [model.NewBoolVar('v'+str(i)) for i in range(len(H))]
    model.Maximize(sum(variables))
    # Add triangle conditions
    #triangles = []
    for i in range(len(H)):
        neighbors = neighboring(i, H) # Neighors connected to vertex
        newNeighbors = np.setdiff1d(neighbors, np.array(range(i)))
        for j in range(len(newNeighbors)):
            for k in range(len(newNeighbors)-j-1):
                if H[neighbors[j]][newNeighbors[j+k+1]] == 1: # Edge connected to vertex
                    #triangles.append({i, newNeighbors[j], newNeighbors[j+k+1})
                    model.AddBoolOr([variables[i].Not(), variables[newNeighbors[j]].Not(), variables[newNeighbors[j+k+1]].Not()])


    # Find feasible cones
    solver = cp_model.CpSolver()
    solution_printer = VarArraySolutionPrinter(variables)
    solver.parameters.enumerate_all_solutions = True
    status = solver.Solve(model, solution_printer)
    print('Status = %s' % solver.StatusName(status))
    print('Number of solutions found: %i' % solution_printer.solution_count())


for graph in r44_14:
    maximalFeasibleCones(graph)
