import parsl
from parsl.app.app import python_app
from parsl.config import Config
from parsl.executors.threads import ThreadPoolExecutor
from parsl.configs.local_threads import config
import networkx as nx

local_threads = Config(
    executors=[
        ThreadPoolExecutor(
            max_threads=2,
            label='local_threads'
        )
    ]
)

parsl.load(local_threads)

@python_app
def glue_16_attempt(graph_obj, a, b, permutation, output_path):
    import networkx as nx
    import itertools
    import GluingClassesJ as gcj

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
        M = gcj.PotentialEdgeMatrix(d_prime, d_prime)
        clauses = []


        # For each (rst)-clique and independent set create a new Clause to represent it
        for clique in rst_cliques:
            g_vertices = clique[1]
            h_vertices = clique[2]
            variables = []
            for g_vertex in g_vertices:
                for h_vertex in h_vertices:
                    variables.append(M.matrix[g_map[g_vertex]][h_map[h_vertex]])
            new_clause = gcj.Clause(variables, gcj.ClauseType.CLIQUE)
            clauses.append(new_clause)

        for indep_set in rst_IS:
            g_vertices = indep_set[1]
            h_vertices = indep_set[2]
            variables = []
            for g_vertex in g_vertices:
                for h_vertex in h_vertices:
                    variables.append(M.matrix[g_map[g_vertex]][h_map[h_vertex]])
            new_clause = gcj.Clause(variables, gcj.ClauseType.INDEP_NO_EDGES)
            clauses.append(new_clause)

        for indep_set_OE in rst_ISOE:
            g_vertices = indep_set_OE[1]
            h_vertices = indep_set_OE[2]
            variables = []
            for g_vertex in g_vertices:
                for h_vertex in h_vertices:
                    variables.append(M.matrix[g_map[g_vertex]][h_map[h_vertex]])
            new_clause = gcj.Clause(variables, gcj.ClauseType.INDEP_SAME_SET_EDGE)
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
            if alpha.exists == gcj.EdgeExists.FALSE:
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
                            if beta.exists == gcj.EdgeExists.UNKNOWN:
                                beta.set_exists(gcj.EdgeExists.TRUE)
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
                            if beta.exists == gcj.EdgeExists.UNKNOWN:
                                beta.set_exists(gcj.EdgeExists.TRUE)
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
                            if beta.exists == gcj.EdgeExists.UNKNOWN:
                                beta.set_exists(gcj.EdgeExists.FALSE)
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
                if clause.clause_type == gcj.ClauseType.CLIQUE:
                    clause.potential_edges[0].set_exists(gcj.EdgeExists.FALSE)
                    stack.append(clause.potential_edges[0])
                elif clause.clause_type == gcj.ClauseType.INDEP_SAME_SET_EDGE:
                    clause.potential_edges[0].set_exists(gcj.EdgeExists.TRUE)
                    stack.append(clause.potential_edges[0])
                else:
                    #in this case, its a (4,1,1) indep set with no edges, but we need 2 edges.
                    print("Problem", clause)
                    return None
            elif len(clause.potential_edges) == 2:
                if clause.clause_type == gcj.ClauseType.INDEP_NO_EDGES:
                    clause.potential_edges[0].set_exists(gcj.EdgeExists.TRUE)
                    stack.append(clause.potential_edges[0])
                    clause.potential_edges[1].set_exists(gcj.EdgeExists.TRUE)
                    stack.append(clause.potential_edges[1])
                    
        return stack

    #Description: Takes the variables and clauses and recursively calls the stack algo to find all gluings
    #Input: M are the initial variables, clauses are the clauses 
    #This probably means we need to make a copy of M/clauses/a new stack for each execution in case of failure.
    #Output: list of possible gluings
    def recursive_solving(M, clauses, stack, depth):
        if stack_algo(stack) == False:
            #print("Failed at depth ", depth)
            return None
        else:
            unknown_list = []
            for list in M.matrix:
                for elem in list:
                    if elem.exists == gcj.EdgeExists.UNKNOWN:
                        unknown_list.append(elem)
            if len(unknown_list) == 0:
                return M
            else:
                #this part of the code creates a copy of M and clauses
                #we use the original M and clauses for the first recursion, and the copy for a second
                #this copying is done to prevent changes in one recursion from affecting the second
                num_rows = len(M.matrix)
                num_cols = len(M.matrix[0])
                M_copy = gcj.PotentialEdgeMatrix(num_rows, num_cols)
                clauses_copy = []
                for clause in clauses:
                    #we only copy unsatisfied clauses as an optimization
                    if not (clause.is_satisfied()):
                        old_vars = clause.potential_edges
                        new_vars = []
                        for var in old_vars:
                            new_vars.append(M_copy.matrix[var.G_vertex][var.H_vertex])
                        new_clause = gcj.Clause(new_vars, clause.clause_type)
                        clauses_copy.append(new_clause)
                        
                for i in range(num_rows):
                    for j in range(num_cols):
                        if M.matrix[i][j].exists == gcj.EdgeExists.TRUE:
                            M_copy.matrix[i][j].set_exists(gcj.EdgeExists.TRUE)
                        elif M.matrix[i][j].exists == gcj.EdgeExists.FALSE:
                            M_copy.matrix[i][j].set_exists(gcj.EdgeExists.FALSE)
                
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
                
                M.matrix[next_vertex.G_vertex][next_vertex.H_vertex].set_exists(gcj.EdgeExists.FALSE)
                M_copy.matrix[next_vertex.G_vertex][next_vertex.H_vertex].set_exists(gcj.EdgeExists.TRUE)
                list1 = recursive_solving(M, clauses, stack, depth+1)           
                if list1 is not None:
                    return list1 
                list2 = recursive_solving(M_copy, clauses_copy, stack_copy, depth+1)
                return list2

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
                if M.matrix[g_map[x]][h_map[y]].exists == gcj.EdgeExists.TRUE:
                    glued_graph.add_edge(x, y)
        
        return glued_graph

    G = graph_obj.copy()
    G = nx.complement(G)
    G_k = list(nx.neighbors(G,a))
    G_k.sort()
    map1 = {a:"a"}
    for i in range(len(G_k)):
        map1[G_k[i]] = "k"+str(i)
    for i in range(16):
        if i not in map1:
            map1[i] = "g"+str(i)
    nx.relabel_nodes(G, map1, False)
    print(G.nodes())

    H = graph_obj.copy()
    H = nx.complement(H)
    print(H.nodes())
    H_k = list(nx.neighbors(H,b))
    H_k.sort()
    map2 = {b:"b"}
    for i in range(len(H_k)):
        map2[H_k[i]] = permutation[i]
    for i in range(16):
        if i not in map2:
            map2[i] = "h"+str(i)
    nx.relabel_nodes(H, map2, False)
    print(H.nodes())

    K = nx.Graph()
    K.add_nodes_from(["k0", "k1", "k2", "k3", "k4"])

    clauses, M, g_map, h_map = create_SAT_formula(G, "a", H, "b", K)
    # sum = 0
    # for i in range(100000000):
    #     sum += 1
    
    # return False 
    stack = setup_stack(clauses)
    
    #print(stack)
    if stack is None:
        print("4,1,1 issue")
        return False 
    else:
        solution = recursive_solving(M, clauses, stack, 0)
        if solution == None:
            print("no solution found...?")
            return False 
        else:
            glued = glue(G, "a", H, "b", solution, g_map, h_map)
            #nx.write_graph6(glued, output_path)
            return True
    
COMMON_GRAPH = nx.read_graph6('dataset_k3kme/k3k6e_16.g6')
gluing_attempts = []
pair = (0,0)
#for perm in itertools.permutations(["k0", "k1", "k2", "k3", "k4"], 5):
perm = ("k0", "k1", "k2", "k3", "k4")
perm_str = ''.join(perm)
output_file = "gluek3k6e_16_" + str(pair[0]) + "_" + str(pair[1]) + "_" + perm_str
gluing_attempts.append(glue_16_attempt(COMMON_GRAPH.copy(), 0, 0, perm, output_file))

perm2 = ("k0", "k1", "k2", "k4", "k3")
perm_str2 = ''.join(perm2)
output_file2 = "gluek3k6e_16_" + str(pair[0]) + "_" + str(pair[1]) + "_" + perm_str2
gluing_attempts.append(glue_16_attempt(COMMON_GRAPH.copy(), 0, 0, perm, output_file2))

outputs = [i.result() for i in gluing_attempts]
print(outputs)

# @python_app
# def pi(num_points):
#     from random import random
    
#     inside = 0   
#     for i in range(num_points):
#         x, y = random(), random()  # Drop a random point in the box.
#         if x**2 + y**2 < 1:        # Count points within the circle.
#             inside += 1
    
#     return (inside*4 / num_points)

# # App that computes the mean of three values
# @python_app
# def mean(a, b, c):
#     return (a + b + c) / 3

# # Estimate three values for pi
# a, b, c = pi(10**8), pi(10**8), pi(10**8)

# # Compute the mean of the three estimates
# mean_pi  = mean(a, b, c)

# # Print the results
# print("a: {:.5f} b: {:.5f} c: {:.5f}".format(a.result(), b.result(), c.result()))
# print("Average: {:.5f}".format(mean_pi.result()))