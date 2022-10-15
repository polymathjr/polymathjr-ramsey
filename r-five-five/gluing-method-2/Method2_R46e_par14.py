import parsl
from parsl.app.app import python_app
from parsl.config import Config
from parsl.executors import HighThroughputExecutor
from parsl.providers import LocalProvider
from parsl.providers import AdHocProvider
from parsl.channels import LocalChannel
from parsl.channels import SSHChannel
import networkx as nx
import itertools
import pickle


local_thing = Config(
    executors=[
        HighThroughputExecutor(
            label="local_htex",
            worker_debug=False,
            max_workers=48,
            provider=LocalProvider(
                channel=LocalChannel(),
                init_blocks=1,
                max_blocks=1,
            ),
        )
    ]
)

parsl.load(local_thing)

@python_app
def glue_14_attempt(graph_obj_1, a, graph_obj_2, b, permutation, output_path):
    import networkx as nx
    import itertools
    import importlib.util
    import sys
    # spec = importlib.util.spec_from_file_location("module.name", "./GluingClassesJ.py")
    # gcj = importlib.util.module_from_spec(spec)
    # sys.modules["module.name"] = gcj
    # spec.loader.exec_module(gcj)
    #foo.MyClass()
    #import GluingClassesJ as gcj
    #PUTTING GLUINGCLASSESJ HERE

    import enum
    class EdgeExists(enum.Enum):
        UNKNOWN = 0 
        TRUE = 1 
        FALSE = 2 

    class ClauseType(enum.Enum):
        CLIQUE = 0 
        INDEP_NO_EDGES = 1 
        INDEP_SAME_SET_EDGE = 2 


    # Define a class to represent variables as described on page 7
    class PotentialEdge:
        
        # Construct a new PotentialEdge between vertex G_vert in G and H_vert in H
        def __init__(self, G_vert, H_vert):
            # Set variables
            self.G_vertex = G_vert
            self.H_vertex = H_vert
            # Set current value of the variable
            self.exists = EdgeExists.UNKNOWN
            # Create sets of clique clauses and independent-set clauses
            self.clique_clauses = []
            self.ind_set_clauses = []
            self.ind_same_set_clauses = []

        def __str__(self):
            if self.exists == EdgeExists.UNKNOWN:
                return "("+str(self.G_vertex) + "," + str(self.H_vertex)+ "," + "U)"
            elif self.exists == EdgeExists.TRUE:
                return "("+str(self.G_vertex) + "," + str(self.H_vertex)+ "," + "T)"
            else:
                return "("+str(self.G_vertex) + "," + str(self.H_vertex)+ "," + "F)"
        
        # Set the value of the variable
        # NOTE: This should only be used when we change the value from UNKNOWN to TRUE or FALSE
        # INPUT: new_value for exists to be set to
        def set_exists(self, new_value):
            self.exists = new_value
            # Decrease the number of unknown for each clause this variable is in
            for clause in self.clique_clauses:
                clause.decr_num_unknown(new_value)
            for clause in self.ind_set_clauses:
                clause.decr_num_unknown(new_value)
            for clause in self.ind_same_set_clauses:
                clause.decr_num_unknown(new_value)
        
        
        # Add a clause that the variable is in
        # INPUT: a clause that this variable is in and that should be added to it's list,
        #        an enum that is the type of clause
        def add_clause(self, clause, clause_type):
            if clause_type == ClauseType.CLIQUE:
                self.clique_clauses.append(clause)
            elif clause_type == ClauseType.INDEP_NO_EDGES:
                self.ind_set_clauses.append(clause)
            else:
                self.ind_same_set_clauses.append(clause)



    # Define a class to represent clauses as described on page 7
    class Clause:
        
        # Construct a new Clause
        # INPUT: a list of PotentialEdges/variables that are contained in this Clause
        #        an enum that is the type of clause
        def __init__(self, variables, clause_type):
            # Set variables
            self.potential_edges = variables
            self.num_unknown = len(variables)
            self.clause_type = clause_type
            # Number of variables whose value is undesired 
            #        (i.e. number of TRUEs if it's a clique clause, FALSEs if an independent set clause)
            self.num_undesired = 0 
            
            # Add clauses to each potential edge
            for pot_edge in variables:
                pot_edge.add_clause(self, clause_type)

        def __str__(self):
            output = ""
            for var in self.potential_edges:
                output += str(var) + ", "
            output += str(self.clause_type)
            return output
        
        
        # Decrease the number of unknowns, called when a PotentialEdge's value goes from UNKNOWN to TRUE or FALSE
        # INPUT: the new value that the PotentialEdge was changed to
        def decr_num_unknown(self, new_value):
            # Decrease the number of unknowns by 1
            self.num_unknown -= 1 
            # Update number of variables with undesired value
            if self.clause_type == ClauseType.CLIQUE and new_value == EdgeExists.TRUE:
                self.num_undesired += 1 
            elif self.clause_type != ClauseType.CLIQUE and new_value == EdgeExists.FALSE:
                self.num_undesired += 1 
        
        
        # Determine whether all PotentialEdges in the clause are set to a value (i.e. are not UNKNOWN)
        # OUTPUT: a boolean, True if there are no UNKNOWNs, False otherwise
        def is_full(self):
            return self.num_unknown == 0 
        
        # Determine whether the clique causes a FAIL state
        #       i.e. if all variables are TRUE when this is a clique clause or all variables FALSE when this is an independent set clause
        # OUTPUT: a boolean, True if in a FAIL state, False otherwise
        def in_fail_state(self):
            if self.clause_type == ClauseType.CLIQUE:
                return self.num_unknown == 0  and self.num_undesired == len(self.potential_edges)
            elif self.clause_type == ClauseType.INDEP_NO_EDGES:
                #if there are k potential edges, and k-1 set to False, then 1 is set to true - complement of this is a J
                return self.num_unknown == 0 and self.num_undesired >= len(self.potential_edges) - 1
            else:
                return self.num_unknown == 0  and self.num_undesired == len(self.potential_edges)

        def is_satisfied(self):
            if self.clause_type == ClauseType.INDEP_NO_EDGES:
                return self.num_undesired + self.num_unknown <= len(self.potential_edges) - 2
            else:
                return self.num_undesired + self.num_unknown <= len(self.potential_edges) - 1


    # Construct a class to represent the matrix of variables
    class PotentialEdgeMatrix:
        
        # Construct a new Matrix of Potential Edges as described in the paper
        # INPUT: num_rows = |VG|-|VK|-1, num_cols = |VH|-|VK|-1
        def __init__(self, num_rows, num_cols):
            # Create the matrix
            # NOTE: Should probably change to a more efficient data structure, since python lists are LinkedLists
            self.matrix = []
            for row in range(0 , num_rows):
                current_row = []
                for col in range(0 , num_cols):
                    current_row.append(PotentialEdge(row, col))
                self.matrix.append(current_row)
        def __str__(self):
            output = "["
            for list in self.matrix:
                output += "["
                for elem in list:
                    output += str(elem)
                    output += ", "
                output += "],"
            output += "]"

            return output


    #I think ur supposed to locally define these functions...
    def is_clique(G, nodes):
        all_edges = True
        for pair in itertools.combinations(nodes, 2):
            if not G.has_edge(pair[0], pair[1]):
                all_edges = False
                break
        
        return all_edges


    # Description: Defintion on page 6
    # Input: G, a (first pointed graph we are gluing) and H, b (second pointed graph we are glueing)
    # K is the shared neighborhood of a and b in both graphs (K is in R(3,5,d)). Note that for this method, we assume
    # K's vertices when found in G and H will have the same labels in all 3 graphs.
    # Output: A list of all the (r, s, t) cliques. 
    def get_rst_cliques(G, a, H, b, K):
        #Contains 3-tuples of the form (w_tuple, x_tuple, y_tuple)
        rst_cliques = []
        VK = K.nodes()
        xs = (set(G.nodes()) - set(VK))
        xs.remove(a)
        ys = (set(H.nodes()) - set(VK))
        ys.remove(b)

        def clique_helper(r, s, t, VK_, xs_, ys_):
            output = []
            if r > 0:
                for w_comb in itertools.combinations(VK_,r):
                    #contains s-tuples that form r+s - independent sets with w in G
                    x_cliques = []
                    #contains t-tuples that form r+t - independent sets with w in H
                    y_cliques = []

                    for x_comb in itertools.combinations(xs_, s):
                        comb = w_comb + x_comb
                        if is_clique(G, comb):
                            x_cliques.append(x_comb)

                    for y_comb in itertools.combinations(ys_, t):
                        comb = w_comb + y_comb
                        if is_clique(H, comb):
                            y_cliques.append(y_comb)

                    for x_comb in x_cliques:
                        for y_comb in y_cliques:
                            output.append( ( w_comb, x_comb, y_comb ) )
                return output
            else:
                #contains s-tuples that form s - independent sets in G
                x_cliques = []
                #contains t-tuples that form t - independent sets in H
                y_cliques = []
                for x_comb in itertools.combinations(xs_, s):
                    if is_clique(G, x_comb):
                        x_cliques.append(x_comb)

                for y_comb in itertools.combinations(ys_, t):
                    if is_clique(H, y_comb):
                        y_cliques.append(y_comb)

                for x_comb in x_cliques:
                    for y_comb in y_cliques:
                        output.append( ( (), x_comb, y_comb ) )
                
                return output


        rst_cliques += clique_helper(0,2,2,VK,xs,ys)

        return rst_cliques

    #Also putting this here
    def is_independent_set(G, nodes):
        no_edges = True
        for pair in itertools.combinations(nodes, 2):
            if G.has_edge(pair[0], pair[1]):
                no_edges = False
                break
        
        return no_edges



    # Description: Defintion on page 6
    # Input: G, a (first pointed graph we are gluing) and H, b (second pointed graph we are glueing)
    # K is the shared neighborhood of a and b in both graphs (K is in R(3,J6,d)). Note that for this method, we assume
    # K's vertices when found in G and H will have the same labels in all 3 graphs.
    # Output: A list of all the (r, s, t) independent sets. 
    def get_rst_independent_sets(G, a, H, b, K):
        #Contains 3-tuples of the form (w_tuple, x_tuple, y_tuple)
        rst_IS = []
        VK = K.nodes()
        xs = (set(G.nodes()) - set(VK))
        xs.remove(a)
        ys = (set(H.nodes()) - set(VK))
        ys.remove(b)

        def indepHelper(r,s,t, VK_, xs_, ys_):
            output = []
            if r > 0:
                for w_comb in itertools.combinations(VK_,r):
                    #contains s-tuples that form r+s - independent sets with w in G
                    x_cliques = []
                    #contains t-tuples that form r+t - independent sets with w in H
                    y_cliques = []

                    for x_comb in itertools.combinations(xs_, s):
                        comb = w_comb + x_comb
                        if is_independent_set(G, comb):
                            x_cliques.append(x_comb)

                    for y_comb in itertools.combinations(ys_, t):
                        comb = w_comb + y_comb
                        if is_independent_set(H, comb):
                            y_cliques.append(y_comb)

                    for x_comb in x_cliques:
                        for y_comb in y_cliques:
                            output.append( ( w_comb, x_comb, y_comb ) )
                return output
            else:
                #contains s-tuples that form s - independent sets in G
                x_cliques = []
                #contains t-tuples that form t - independent sets in H
                y_cliques = []
                for x_comb in itertools.combinations(xs_, s):
                    if is_independent_set(G, x_comb):
                        x_cliques.append(x_comb)

                for y_comb in itertools.combinations(ys_, t):
                    if is_independent_set(H, y_comb):
                        y_cliques.append(y_comb)

                for x_comb in x_cliques:
                    for y_comb in y_cliques:
                        output.append( ( (), x_comb, y_comb ) )
                
                return output

        rst_IS += indepHelper(4, 1, 1, VK, xs, ys)
        rst_IS += indepHelper(3, 1, 2, VK, xs, ys)
        rst_IS += indepHelper(3, 2, 1, VK, xs, ys)
        rst_IS += indepHelper(2, 3, 1, VK, xs, ys)
        rst_IS += indepHelper(2, 1, 3, VK, xs, ys)
        rst_IS += indepHelper(2, 2, 2, VK, xs, ys)
        rst_IS += indepHelper(1, 2, 3, VK, xs, ys)
        rst_IS += indepHelper(1, 3, 2, VK, xs, ys)
        rst_IS += indepHelper(1, 4, 1, VK, xs, ys)
        rst_IS += indepHelper(1, 1, 4, VK, xs, ys)
        rst_IS += indepHelper(0, 3, 3, VK, xs, ys)
        rst_IS += indepHelper(0, 2, 4, VK, xs, ys)
        rst_IS += indepHelper(0, 4, 2, VK, xs, ys)
        
        return rst_IS

    # Description: Indep-set but now we can have 1 edge between vertices in G/H.
    # Input: G, a (first pointed graph we are gluing) and H, b (second pointed graph we are glueing)
    # K is the shared neighborhood of a and b in both graphs (K is in R(3,J6,d)). Note that for this method, we assume
    # K's vertices when found in G and H will have the same labels in all 3 graphs.
    # Output: A list of all sort of (r, s, t) independent sets with one edge. 
    def get_rst_indep_sets_one_edge(G, a, H, b, K):
        #Contains 3-tuples of the form (w_tuple, x_tuple, y_tuple)
        rst_IS = []
        VK = K.nodes()
        xs = (set(G.nodes()) - set(VK))
        xs.remove(a)
        ys = (set(H.nodes()) - set(VK))
        ys.remove(b)

        def indepOneEdgeHelper(r,s,t, VK_, xs_, ys_):
            output = []
            if r > 0:
                for w_comb in itertools.combinations(VK_,r):
                    #contains s-tuples that form r+s - independent sets with w in G
                    x_edge_set = []
                    x_no_edge_set = []
                    #contains t-tuples that form r+t - independent sets with w in H
                    y_edge_set = []
                    y_no_edge_set = []
                    
                    for x_comb in itertools.combinations(xs_, s):
                        comb = w_comb + x_comb
                        num_edges = 0
                        for pairs in itertools.combinations(comb, 2):
                            if G.has_edge(pairs[0], pairs[1]):
                                num_edges += 1
                        if num_edges == 1:
                            x_edge_set.append(x_comb)

                    for y_comb in itertools.combinations(ys_, t):
                        comb = w_comb + y_comb
                        if is_independent_set(H, comb):
                            y_no_edge_set.append(y_comb)
                    
                    for x_comb in itertools.combinations(xs_, s):
                        comb = w_comb + x_comb
                        if is_independent_set(G, comb):
                            x_no_edge_set.append(x_comb)

                    for y_comb in itertools.combinations(ys_, t):
                        comb = w_comb + y_comb
                        num_edges = 0
                        for pairs in itertools.combinations(comb, 2):
                            if H.has_edge(pairs[0], pairs[1]):
                                num_edges += 1
                        if num_edges == 1:
                            y_edge_set.append(y_comb)

                    for x_comb in x_edge_set:
                        for y_comb in y_no_edge_set:
                            output.append( ( w_comb, x_comb, y_comb ) )
                    for x_comb in x_no_edge_set:
                        for y_comb in y_edge_set:
                            output.append( ( w_comb, x_comb, y_comb ) )
                return output
            else:
                #contains s-tuples that form r+s - independent sets with w in G
                x_edge_set = []
                x_no_edge_set = []
                #contains t-tuples that form r+t - independent sets with w in H
                y_edge_set = []
                y_no_edge_set = []

                for x_comb in itertools.combinations(xs_, s):
                    num_edges = 0
                    for pairs in itertools.combinations(x_comb, 2):
                        if G.has_edge(pairs[0], pairs[1]):
                            num_edges += 1
                    if num_edges == 1:
                        x_edge_set.append(x_comb)

                for y_comb in itertools.combinations(ys_, t):
                    if is_independent_set(H, y_comb):
                        y_no_edge_set.append(y_comb)

                for x_comb in itertools.combinations(xs_, s):
                    if is_independent_set(G, x_comb):
                        x_no_edge_set.append(x_comb)

                for y_comb in itertools.combinations(ys_, t):
                    num_edges = 0
                    for pairs in itertools.combinations(y_comb, 2):
                        if H.has_edge(pairs[0], pairs[1]):
                            num_edges += 1
                    if num_edges == 1:
                        y_edge_set.append(y_comb)

                for x_comb in x_edge_set:
                    for y_comb in y_no_edge_set:
                        output.append( ( (), x_comb, y_comb ) )
                for x_comb in x_no_edge_set:
                    for y_comb in y_edge_set:
                        output.append( ( (), x_comb, y_comb ) )
                
                return output

        rst_IS += indepOneEdgeHelper(4, 1, 1, VK, xs, ys)
        rst_IS += indepOneEdgeHelper(3, 1, 2, VK, xs, ys)
        rst_IS += indepOneEdgeHelper(3, 2, 1, VK, xs, ys)
        rst_IS += indepOneEdgeHelper(2, 3, 1, VK, xs, ys)
        rst_IS += indepOneEdgeHelper(2, 1, 3, VK, xs, ys)
        rst_IS += indepOneEdgeHelper(2, 2, 2, VK, xs, ys)
        rst_IS += indepOneEdgeHelper(1, 2, 3, VK, xs, ys)
        rst_IS += indepOneEdgeHelper(1, 3, 2, VK, xs, ys)
        rst_IS += indepOneEdgeHelper(1, 4, 1, VK, xs, ys)
        rst_IS += indepOneEdgeHelper(1, 1, 4, VK, xs, ys)
        rst_IS += indepOneEdgeHelper(0, 3, 3, VK, xs, ys)
        rst_IS += indepOneEdgeHelper(0, 2, 4, VK, xs, ys)
        rst_IS += indepOneEdgeHelper(0, 4, 2, VK, xs, ys)
        
        return rst_IS

    def create_SAT_formula(G, a, H, b, K):
        
        # Find the rst-cliques and independent graphs
        rst_cliques = get_rst_cliques(G, a, H, b, K)
        rst_IS = get_rst_independent_sets(G, a, H, b, K)
        rst_ISOE =  get_rst_indep_sets_one_edge(G, a, H, b, K)
        
        VK = K.nodes()
        xs = (set(G.nodes()) - set(VK))
        xs.remove(a)
        g_map = {}
        i = 0
        for x in xs:
            g_map[x] = i
            i += 1

        ys = (set(H.nodes()) - set(VK))
        ys.remove(b)
        h_map = {}
        i = 0
        for y in ys:
            h_map[y] = i
            i += 1

        d_prime = len(xs) #i think this should be correct
        M = PotentialEdgeMatrix(d_prime, d_prime)
        clauses = []


        # For each (rst)-clique and independent set create a new Clause to represent it
        for clique in rst_cliques:
            g_vertices = clique[1]
            h_vertices = clique[2]
            variables = []
            for g_vertex in g_vertices:
                for h_vertex in h_vertices:
                    variables.append(M.matrix[g_map[g_vertex]][h_map[h_vertex]])
            new_clause = Clause(variables, ClauseType.CLIQUE)
            clauses.append(new_clause)

        for indep_set in rst_IS:
            g_vertices = indep_set[1]
            h_vertices = indep_set[2]
            variables = []
            for g_vertex in g_vertices:
                for h_vertex in h_vertices:
                    variables.append(M.matrix[g_map[g_vertex]][h_map[h_vertex]])
            new_clause = Clause(variables, ClauseType.INDEP_NO_EDGES)
            clauses.append(new_clause)

        for indep_set_OE in rst_ISOE:
            g_vertices = indep_set_OE[1]
            h_vertices = indep_set_OE[2]
            variables = []
            for g_vertex in g_vertices:
                for h_vertex in h_vertices:
                    variables.append(M.matrix[g_map[g_vertex]][h_map[h_vertex]])
            new_clause = Clause(variables, ClauseType.INDEP_SAME_SET_EDGE)
            clauses.append(new_clause)
        
        return clauses, M, g_map, h_map

    #Description: Takes a pre-setup stack and computes as much assignment as possible
    #Input: stack is just a list used as a stack of variables. Note that because of how references work in python - modifying parts of stack
    #effects the matrix and clauses outside of this function. 
    #This probably means we need to make a copy of M/clauses/a new stack for each execution in case of failure.
    #Output: Returns true/false if it works
    def stack_algo(stack):
        assignments = []
        while len(stack) > 0:
            alpha = stack.pop()
            if alpha.exists == EdgeExists.FALSE:
                for i_clause in alpha.ind_set_clauses:
                    num_variables = len(i_clause.potential_edges)
                    #This means all are FALSE
                    if i_clause.in_fail_state():
                        #print(str(i_clause) + " isn't satisfiable")
                        return False
                    #this means all are FALSE except two variables
                    #and either 1 is true and 1 is unknown or both are unknown
                    elif i_clause.num_undesired == num_variables - 2 and i_clause.num_unknown >= 1:
                        for beta in i_clause.potential_edges:
                            if beta.exists == EdgeExists.UNKNOWN:
                                beta.set_exists(EdgeExists.TRUE)
                                #print(str(beta) + " is now true")
                                stack.append(beta)
                for i_clause in alpha.ind_same_set_clauses:
                    num_variables = len(i_clause.potential_edges)
                    #This means all are FALSE
                    if i_clause.in_fail_state():
                        #print(str(i_clause) + " isn't satisfiable")
                        return False
                    #this means all are FALSE except one variable
                    elif i_clause.num_undesired == num_variables - 1 and i_clause.num_unknown == 1:
                        for beta in i_clause.potential_edges:
                            if beta.exists == EdgeExists.UNKNOWN:
                                beta.set_exists(EdgeExists.TRUE)
                                #print(str(beta) + " is now true")
                                stack.append(beta)
            else:
                for c_clause in alpha.clique_clauses:
                    num_variables = len(c_clause.potential_edges)
                    #this means all are TRUE
                    if c_clause.in_fail_state():
                        #print(str(c_clause) + " isn't satisfiable")
                        return False
                    #this means all are TRUE except one variable
                    elif c_clause.num_undesired == num_variables - 1 and c_clause.num_unknown == 1:
                        for beta in c_clause.potential_edges:
                            if beta.exists == EdgeExists.UNKNOWN:
                                beta.set_exists(EdgeExists.FALSE)
                                #print(str(beta) + " is now false")
                                stack.append(beta)
        return True
    
    # Sets up a stack to be used in the first call to stack_algo()
    # INPUT: The set of clauses
    # OUTPUT: A stack set up to be used in a first call to stack_algo()
    # or None if no stack can be set up
    def setup_stack(clauses):
        stack = []
        
        for clause in clauses:
            if len(clause.potential_edges) == 1:
                if clause.clause_type == ClauseType.CLIQUE:
                    clause.potential_edges[0].set_exists(EdgeExists.FALSE)
                    stack.append(clause.potential_edges[0])
                elif clause.clause_type == ClauseType.INDEP_SAME_SET_EDGE:
                    clause.potential_edges[0].set_exists(EdgeExists.TRUE)
                    stack.append(clause.potential_edges[0])
                else:
                    #in this case, its a (4,1,1) indep set with no edges, but we need 2 edges.
                    print("Problem", clause)
                    return None
            elif len(clause.potential_edges) == 2:
                if clause.clause_type == ClauseType.INDEP_NO_EDGES:
                    clause.potential_edges[0].set_exists(EdgeExists.TRUE)
                    stack.append(clause.potential_edges[0])
                    clause.potential_edges[1].set_exists(EdgeExists.TRUE)
                    stack.append(clause.potential_edges[1])
                    
        return stack

    #Description: Takes the variables and clauses and recursively calls the stack algo to find all gluings
    #Input: M are the initial variables, clauses are the clauses 
    #This probably means we need to make a copy of M/clauses/a new stack for each execution in case of failure.
    #Output: list of possible gluings
    def recursive_solving(M, clauses, stack, G, a, H, b, g_map, h_map, file_to_write):
        #print(depth)
        if stack_algo(stack) == False:
            #print("Failed at depth ", depth)
            return 0
            #return False
        else:
            unknown_list = []
            for list in M.matrix:
                for elem in list:
                    if elem.exists == EdgeExists.UNKNOWN:
                        unknown_list.append(elem)
            if len(unknown_list) == 0:
                glued = glue(G, a, H, b, M, g_map, h_map)
                g6_bytes = nx.to_graph6_bytes(glued)
                file_to_write.write(g6_bytes)
                return 1
                #return True
            else:
                #this part of the code creates a copy of M and clauses
                #we use the original M and clauses for the first recursion, and the copy for a second
                #this copying is done to prevent changes in one recursion from affecting the second
                num_rows = len(M.matrix)
                num_cols = len(M.matrix[0])
                M_copy = PotentialEdgeMatrix(num_rows, num_cols)
                clauses_copy = []
 
                for clause in clauses:
                    #we only copy unsatisfied clauses as an optimization
                    if not (clause.is_satisfied()):
                        old_vars = clause.potential_edges
                        new_vars = []
                        for var in old_vars:
                            new_vars.append(M_copy.matrix[var.G_vertex][var.H_vertex])
                        new_clause = Clause(new_vars, clause.clause_type)
                        clauses_copy.append(new_clause)
                        
                for i in range(num_rows):
                    for j in range(num_cols):
                        if M.matrix[i][j].exists == EdgeExists.TRUE:
                            M_copy.matrix[i][j].set_exists(EdgeExists.TRUE)
                        elif M.matrix[i][j].exists == EdgeExists.FALSE:
                            M_copy.matrix[i][j].set_exists(EdgeExists.FALSE)
                
                #Using the paper's heuristic to decide what variable to add to the stack
                next_vertex = unknown_list[0]
                best_forcing = 0
                for unknown_candidate in unknown_list:
                    num_forcing = 0
                    for c_clause in unknown_candidate.clique_clauses:
                        num_variables = len(c_clause.potential_edges)
                        if c_clause.num_unknown == 2 and c_clause.num_undesired == num_variables - 2:
                            num_forcing += 1 
                    for i_clause in unknown_candidate.ind_same_set_clauses:
                        num_variables = len(i_clause.potential_edges)
                        if i_clause.num_unknown == 2 and i_clause.num_undesired == num_variables - 2:
                            num_forcing += 1 
                    for i_clause in unknown_candidate.ind_set_clauses:
                        num_variables = len(i_clause.potential_edges)
                        if i_clause.num_unknown == 3 and i_clause.num_undesired == num_variables - 3:
                            num_forcing += 1 
                    if num_forcing > best_forcing:
                        next_vertex = unknown_candidate
                        best_forcing = num_forcing
                
                stack = [next_vertex]
                stack_copy = [M_copy.matrix[next_vertex.G_vertex][next_vertex.H_vertex]]
                
                M.matrix[next_vertex.G_vertex][next_vertex.H_vertex].set_exists(EdgeExists.FALSE)
                M_copy.matrix[next_vertex.G_vertex][next_vertex.H_vertex].set_exists(EdgeExists.TRUE)
                val1 = recursive_solving(M, clauses, stack, G, a, H, b, g_map, h_map, file_to_write)           
                val2 = recursive_solving(M_copy, clauses_copy, stack_copy, G, a, H, b, g_map, h_map, file_to_write)
                return val1 + val2

    # Description: We will use a matrix M to glue two pointed graphs together. 
    # Input: Two pointed graphs (G, a) and (H, b) as well as the intersection K. 
    #        Also a d' x d' matrix M and g_map, h_map mappings from the matrix to the vertices of G and H. 
    # Output: A graph, based on glueing along M. 
    def glue(G, a, H, b, M, g_map, h_map): 
        # Copy G and H into a new graph
        glued_graph = G.copy()
        glued_graph.add_nodes_from(H.nodes())
        glued_graph.add_edges_from(H.edges())
        
        # Connect b to all vertices in G and a to all vertices in H
        for g_vertex in G.nodes():
            glued_graph.add_edge(b, g_vertex)
            
        for h_vertex in H.nodes():
            glued_graph.add_edge(a, h_vertex)
        
        # Add edges between vertices of G and H based on a succesfull gluing represented by M
        for x in g_map:
            for y in h_map:
                if M.matrix[g_map[x]][h_map[y]].exists == EdgeExists.TRUE:
                    glued_graph.add_edge(x, y)
        
        return glued_graph

    G = graph_obj_1.copy()
    G_k = list(nx.neighbors(G,a))
    G_k.sort()
    map1 = {a:"a"}
    for i in range(len(G_k)):
        map1[G_k[i]] = "k"+str(i)
    for i in range(14):
        if i not in map1:
            map1[i] = "g"+str(i)
    nx.relabel_nodes(G, map1, False)
    #print(G.nodes())

    H = graph_obj_2.copy()
    H_k = list(nx.neighbors(H,b))
    H_k.sort()
    map2 = {b:"b"}
    for i in range(len(H_k)):
        map2[H_k[i]] = permutation[i]
    for i in range(14):
        if i not in map2:
            map2[i] = "h"+str(i)
    nx.relabel_nodes(H, map2, False)
    #print(H.nodes())

    K = nx.Graph()
    if len(permutation) == 5:
        K.add_nodes_from(["k0", "k1", "k2", "k3", "k4"])
    elif len(permutation) == 4:
        K.add_nodes_from(["k0", "k1", "k2", "k3"])
    else:
        K.add_nodes_from(["k0", "k1", "k2"])

    clauses, M, g_map, h_map = create_SAT_formula(G, "a", H, "b", K)
    stack = setup_stack(clauses) 
    #print(stack)
    if stack is None:
        print("4,1,1 issue")
        return 0
    else:
        output_file = open(output_path, "wb")
        solution = recursive_solving(M, clauses, stack, G, "a", H, "b", g_map, h_map, output_file)
        output_file.close()
        return solution
    
graph_list = nx.read_graph6('dataset_k3kme/k3k6e_14.g6')
gluing_attempts = {}
file_to_read = open("14gluingproblems", "rb")
gluing_problems = pickle.load(file_to_read)
file_to_read.close()

index = 0
for problem in gluing_problems:
    graph1 = graph_list[problem[0][0]]
    graph1 = nx.complement(graph1)
    graph2 = graph_list[problem[1][0]]
    graph2 = nx.complement(graph2)
    a = problem[0][1]
    b = problem[1][1]
    perm = problem[2]
    output_path = "14-graphs/14iso"+str(index)+".g6"
    index += 1
    gluing_attempts[((problem[0][0], problem[0][1]), (problem[1][0], problem[1][1]), perm)] = glue_14_attempt(graph1, a, graph2, b, perm, output_path)
        

outputs = {}
for pair in gluing_attempts:
    #print(pair)
    #print(gluing_attempts[pair].result())
    outputs[pair] = gluing_attempts[pair].result()
    print(str(pair) + ", " + str(outputs[pair]))