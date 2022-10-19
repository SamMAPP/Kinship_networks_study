import os
import random as rd
import numpy as np
import math
import networkx as nx
import matplotlib.pyplot as plt
import collections
import time


def load_distr(filename, has_headers=True):
    """
    Function that uploads the probability distribution and returns a dictionary with it.
    Archive containing the information must have the structure:

    n       probability
    (int)   (float)
    ....    ......

    Output is in the form:
    distr[n]=p(n) (given n as key, returns the probability of having it)
    """
    distr = {}
    types = [int, float]
    with open(filename, "rt") as rows:
        if has_headers:
            headers = next(rows)
            for line in rows:
                line = line.split("\t")
                n_p = ["", ""]
                try:
                    n_p = [func(val) for func, val in zip(types, line)]
                except:
                    n_p[0] = types[0](float(line[0]))
                    n_p[1] = types[1](float(line[1][:-1]))
                distr[n_p[0]] = n_p[1]
        else:
            for line in rows:

                line = line.split("\t")
                # print(line)
                n_p = ["", ""]
                try:
                    n_p = [func(val) for func, val in zip(types, line)]
                except:
                    # print(line)
                    n_p[0] = types[0](float(line[0]))
                    n_p[1] = types[1](float(line[1][:-1]))
                    # print(n_p)
                distr[n_p[0]] = n_p[1]

    return distr


def acumulate_prob_dict(given_dict):
    """
    Returns a dictionary with the cummulative probability distribution of a given PDF d[n]=f_n:
    A[n]=sum of all d[i] with i<=n
    n must be type integer
    probabilities must be type float

    """
    A = {}
    suma = 0
    for n in given_dict.keys():
        suma += given_dict[n]
        A[n] = suma
    return A


def linear_search_list(L, x):
    """
    Given a sorted list L, returns the position where the element x is using a linear search.
    If x is not L, returns the position where it must be inserted.
    """

    for i, z in enumerate(L):
        if x <= z:
            return i


def binary_search_list(L, x):
    pos = -1
    left = 0
    right = len(L) - 1
    while left <= right:
        middle = (left + right) // 2
        if L[middle] == x:
            pos = middle
            break

        if L[middle] > x:
            right = middle - 1
        else:
            left = middle + 1

    if not (left <= right):
        pos = left

    return pos


def create_F1(N, f_n, acum):
    """
    Creates a population of M couples. Each copules has n children with probability f_n.
    Returns a dictionary containing the list of children per couple

    Output
    parent_couple :  [children .. ..]       (dict)
    children : parent_couple ID             (dict)
    M                                       (int)


    """

    # M is the number of couples that we want
    # f_n is the probability distribution that we want those couples to have

    # parents couples to children list contains the information of   couple_ID: [child_ID, child_ID]
    pc2cl = {}

    # c2p[i]: m_i   indicates at which parent couple is linked the chidl i
    c2p = {}

    acum_list_values = [acum[k] for k in list(acum.keys())]
    acum_list_keys = [int(k) for k in list(acum.keys())]

    child_ID_counter = 0

    N_dyn = 0
    M_dyn = 0

    while N_dyn < 2 * (N):
        # generate a random number 'r' between 0 and 1
        r = rd.random()

        # find out where 'r' is in the accumulate curve, and that gives us the number 'n' of children added to couple i
        k = binary_search_list(acum_list_values, r)

        while k == len(acum_list_keys):
            r = rd.random()
            k = binary_search_list(acum_list_values, r)

        n = acum_list_keys[k]
        N_dyn += n

        if N_dyn > 2 * (N):
            n -= N_dyn - 2 * (N)
            N_dyn = 2 * N

        if n:
            for aux in range(n):
                try:
                    pc2cl[M_dyn].append(child_ID_counter)

                except:
                    pc2cl[M_dyn] = [child_ID_counter]

                c2p[child_ID_counter] = M_dyn
                child_ID_counter += 1

        elif n == 0:
            pc2cl[M_dyn] = []

        M_dyn += 1
    # print('Ndyn=', N_dyn)
    # print('M_dyn=', M_dyn)
    # print('max_ID=', child_ID_counter)

    return pc2cl, c2p, M_dyn


def marriages_F1(N, p2c, c2p):
    """
    Function that given a population, assigns marriages within individuals.
    Rules:
    - an individual can only marry once
    - marriages within siblings is forbidden

    Note: When a pair is formed, the individuals in question are assigned randomly chosen opposite sexes.
    Input:
    - N : (int) size of the population
    - alpha: growing factor
    """

    # make a list of all people in F1 and reverse the p2c
    # (make a dictionary of child: parent_couple)
    F1_population = [i for i in range(2 * N)]

    spouse = {}
    gender = {}

    count = 0
    sucess = 1
    while len(F1_population) != 0 and count < 10 * N * 2:
        ok = 0
        count = 0

        while ok == 0:
            # pick 2 different individuals x and y from the F1 population
            x, y = rd.choices(F1_population, k=2)
            count += 1
            # condition: y must not be of the same gender, and not a blood sibling
            if c2p[x] != c2p[y]:
                spouse[x] = y
                spouse[y] = x
                gender[x] = "m"
                gender[y] = "f"
                # add x and y to the list of engaded people
                F1_population.pop(binary_search_list(F1_population, x))
                F1_population.pop(binary_search_list(F1_population, y))
                # engaged.append(x)
                # engaged.append(y)
                ok = 1

            if count >= 10 * N * 2:
                sucess = 0
                break

    del count, ok

    return sucess, spouse, gender


def generate_network(N, f_n, acum):

    """
    Creates a kinship network of size 2*N.
    The f(n) function is the probability that a couple of parents has 'n' children.

    Marriages are assigned fully random between nodes in population of children if:
        -the nodes are not blood siblings (brother-sister)
        -the nodes has different gender (heterosexual couples mating)
        -both nodes are single (monogamous couples)
    Gender is asigned fully at random to each person (male and female only) during mating.

    Input:
            -number of people of each gender in children population N
            -f(n)
            -cummulative probability distribution of f(n)

    Output:
            Returns a Graph (the kinship network of the population).

            Nodes are the members of the population of both genders

            Edges are builded by chossing:
                -in blood relations (brothers and sisters of a node)
                -marriage relation (spouse of the node)

    """

    # Creating the F1 generation

    p2c, c2p, M = create_F1(N, f_n, acum)

    del M

    # Assignation of random marriages

    x = 0

    sucess, spouses_dict, gender = marriages_F1(N, p2c, c2p)

    # print(len(spouses_dict.keys()))

    while not sucess:
        x += 1
        p2c, c2p, M = create_F1(N, f_n, acum)
        sucess, spouses_dict, gender = marriages_F1(N, p2c, c2p)
        # print('Hizo falta volver a asignar matrimonios')

    del x

    ######################Build graph #################
    G = nx.Graph()
    popul_F1 = list(p2c.keys())
    # print(popul_F1)
    G.add_nodes_from([i for i in range(2 * N)])  # Add nodes

    # Add blood siblings edges
    for e in list(p2c.keys()):
        L = p2c[e]

        for i in range(0, len(L) - 1):
            for j in range(i + 1, len(L)):
                G.add_edge(L[i], L[j])

    # Add marriage links
    # print(gender)
    gdr = "m"
    for p in range(2 * N):
        # print(p, 'gender[p]==gdr?', gender[p]==gdr)
        if gender[p] == gdr:
            G.add_edge(p, spouses_dict[p])

    # print(nx.number_of_nodes(G))

    return G


def randomize_net(G, times_randomization=3):
    """
    Randomizes a given network, by selecting two edges at random and exchanging their nodes.
    This process is made a number times_randomization* number_edges.
    This method keeps the degree distribution of network

    Input:  Network  (type: Graph)

    Output:  Random counterpart of the network (type: Graph)

    """

    # get nodes and edges
    n = G.nodes()
    edges = G.edges()

    list_edges = [e for e in edges]

    for i in range(times_randomization * len(list_edges)):
        ind_1, ind_2 = rd.choices(range(len(list_edges)), k=2)
        helper = list_edges[ind_1][1]

        list_edges[ind_1] = (list_edges[ind_1][0], list_edges[ind_2][1])
        list_edges[ind_2] = (list_edges[ind_2][0], helper)

    r = nx.Graph()
    r.add_nodes_from(n)
    r.add_edges_from(list_edges)

    return r


def export_edges_to_file(G, filename):
    """
    Export the edges in graph  'G'   to a file named 'filename'

    """

    edges = list(G.edges())
    f = open(filename, "w")
    for tup in edges:
        f.write("%d %d\n" % (tup[0], tup[1]))
    f.close()


###Experiments with the exponential
def analyze_CCs(
    N,
    iters,
    alfa_0,
    number_points,
    step_alfa,
    n_max,
    start_zero=True,
    show_plots=False,
    compute_nCCs=True,
):

    # Initialization of results containers
    if compute_nCCs:
        CC_number = np.zeros(
            (2, number_points)
        )  # Container for mean value and standart deviation

    CC_sizes = []  # Conatiner for probability distribution functions

    counter = 0

    print("###################  Starting CC   ###################")
    for i in range(number_points):

        # Update the value of 'alfa'
        alfa = alfa_0 + i * step_alfa

        ##################Definition of the f(n) distribution###########################
        f_n = {}

        if start_zero:
            a = 1 + 1 / (2 * alfa)

            f_n = {i: (1 - 1 / a) * a ** (-i) for i in range(0, n_max)}
            acum = acumulate_prob_dict(f_n)

        elif not start_zero:
            a = 2 * alfa / (2 * alfa - 1)
            f_n = {i: (a - 1) * a ** (-i) for i in range(1, n_max)}
            acum = acumulate_prob_dict(f_n)

        ########################Create population##########################################

        ccsizes = np.array([])
        if compute_nCCs:
            ccnumber = np.array([])
        # Do this several times for
        for j in range(iters):
            counter += 1
            if 100 * counter % (iters * number_points) == 0:
                print("Program at %d  done" % (100 * counter / (iters * number_points)))

            # Generate the network
            G = generate_network(N, f_n, acum)

            # Analysis of graph
            if compute_nCCs:
                ncc = nx.algorithms.components.number_connected_components(G)

            scc = np.array([len(r) for r in nx.connected_components(G)])

            # Save the results
            if compute_nCCs:
                ccnumber = np.append(ccnumber, ncc)
            ccsizes = np.append(ccsizes, scc)

        # Save the moments of the distribution
        if compute_nCCs:
            CC_number[0][i] = np.mean(ccnumber)
            CC_number[1][i] = np.std(ccnumber)

        # Create a degree distribution for that value of 'a'
        c = collections.Counter(ccsizes)
        # print('para <n>=',1/(a-1), 'componentes:', ccsizes)
        l_keys = np.array(list(c.keys()))
        l_keys = np.sort(l_keys)

        # Normalization of counts
        tot = 0
        for r in l_keys:
            tot += c[r]

        for r in l_keys:
            c[r] = c[r] / tot

        # Saving distribution
        CC_sizes.append(c)
        seud_name = int(alfa * 1000)
        name = "CC_sizes_%d.txt" % seud_name
        # name='Pruebita.txt'
        documento = open(name, "w")
        for x in l_keys:
            # print((int(x), c[x]))
            documento.write("%d\t%f\n" % (int(x), c[x]))
        documento.close()

    if compute_nCCs:
        # Export mean VD
        f = open("CC_moments.txt", "w")

        for i in range(number_points):
            alfa = alfa_0 + i * step_alfa
            f.write("%f\t%f\t%f\n" % (alfa, CC_number[0][i], CC_number[1][i]))

        f.close()

    if show_plots:
        visualize = number_points // 2
        x = np.array(list(CC_sizes[visualize].keys()))
        x = np.sort(x)
        y = np.array([CC_sizes[visualize][elem] for elem in x])
        plt.scatter(x, y)
        plt.ylabel("probability density")
        plt.yscale("log")
        plt.xlabel("size of CC")
        plt.show()


def analyze_VD(
    N, iters, alfa_0, number_points, step_alfa, n_max, start_zero=True, show_plots=False
):
    # Initialization of parameters

    alfa_0 = 0.1
    n_max = 25

    number_points = 39
    step_alfa = 0.05

    # Initialization of results containers
    vd_moments = np.zeros((2, number_points))
    vd_distr = []

    counter = 0

    print("###################  Starting VD   ###################")
    for i in range(number_points):

        # Update the value of 'alfa'
        alfa = alfa_0 + i * step_alfa

        ##################Definition of the f(n) distribution###########################
        f_n = {}

        if start_zero:
            a = 1 + 1 / (2 * alfa)

            f_n = {i: (1 - 1 / a) * a ** (-i) for i in range(0, n_max)}
            acum = acumulate_prob_dict(f_n)

        elif not start_zero:
            a = 2 * alfa / (2 * alfa - 1)
            f_n = {i: (a - 1) * a ** (-i) for i in range(1, n_max)}
            acum = acumulate_prob_dict(f_n)

        ########################Create population##########################################

        k = np.array([])
        # Do this several times for
        for j in range(iters):
            counter += 1
            if 100 * counter % (iters * number_points) == 0:
                print("Program at %d  done" % (100 * counter / (iters * number_points)))

            # Generate the network
            G = generate_network(N, f_n, acum)

            # Analysis of graph
            ks = np.array(nx.degree(G))[:, 1]

            # Save the results
            k = np.append(k, ks)

        # Save the moments of the distribution
        vd_moments[0][i] = np.mean(k)
        vd_moments[1][i] = np.std(k)

        # Create a degree distribution for that value of 'a'
        c = collections.Counter(k)
        l_keys = np.array(list(c.keys()))
        l_keys = np.sort(l_keys)

        # Normalization of counts
        tot = 0
        for r in l_keys:
            tot += c[r]

        for r in l_keys:
            c[r] = c[r] / tot

        # Saving distribution
        vd_distr.append(c)
        seud_name = int(alfa * 100)
        name = "VD_distr_%d.txt" % seud_name
        # name='Pruebita.txt'
        documento = open(name, "w")
        for x in l_keys:
            # print((int(x), c[x]))
            documento.write("%d\t%f\n" % (int(x), c[x]))
        documento.close()

    # Export mean VD
    f = open("VD_moments.txt", "w")

    for i in range(number_points):
        alfa = alfa_0 + i * step_alfa
        f.write("%f\t%f\t%f\n" % (alfa, vd_moments[0][i], vd_moments[1][i]))

    f.close()

    if show_plots:
        visualize = number_points // 2
        x = np.array(list(vd_distr[visualize].keys()))
        x = np.sort(x)
        y = np.array([vd_distr[visualize][elem] for elem in x])
        plt.scatter(x, y)
        plt.ylabel("probability density")
        plt.xlabel("k")
        plt.show()


def analyze_micros(
    N, iters, alfa_0, number_points, step_alfa, n_max, start_zero=True, show_plots=False
):
    # Initialization of parameters

    alfa_0 = 0.1
    n_max = 25

    number_points = 39
    step_alfa = 0.05

    # Initialization of results containers
    assortativity = np.zeros((2, number_points))
    MCC = np.zeros((2, number_points))
    GCC = np.zeros((2, number_points))

    counter = 0

    print("###################  Starting micros   ###################")
    for i in range(number_points):

        # Update the value of 'alfa'
        alfa = alfa_0 + i * step_alfa

        ##################Definition of the f(n) distribution###########################
        f_n = {}

        if start_zero:
            a = 1 + 1 / (2 * alfa)

            f_n = {i: (1 - 1 / a) * a ** (-i) for i in range(0, n_max)}
            acum = acumulate_prob_dict(f_n)

        elif not start_zero:
            a = 2 * alfa / (2 * alfa - 1)
            f_n = {i: (a - 1) * a ** (-i) for i in range(1, n_max)}
            acum = acumulate_prob_dict(f_n)

        ########################Create population##########################################

        aux_MCC = np.array([])
        aux_GCC = np.array([])
        aux_assort = np.array([])

        # Do this several times for
        for j in range(iters):
            counter += 1
            if 100 * counter % (iters * number_points) == 0:
                print("Program at %d  done" % (100 * counter / (iters * number_points)))

            # Generate the network
            G = generate_network(N, f_n, acum)

            # Analysis of graph

            # Save the results
            aux_MCC = np.append(aux_MCC, nx.algorithms.cluster.average_clustering(G))
            aux_GCC = np.append(aux_GCC, nx.algorithms.cluster.transitivity(G))
            aux_assort = np.append(
                aux_assort,
                nx.algorithms.assortativity.degree_pearson_correlation_coefficient(G),
            )

        # Save the moments of the distribution
        assortativity[0][i] = np.mean(aux_assort)
        assortativity[1][i] = np.std(aux_assort)
        MCC[0][i] = np.mean(aux_MCC)
        MCC[1][i] = np.std(aux_MCC)
        GCC[0][i] = np.mean(aux_GCC)
        GCC[1][i] = np.std(aux_GCC)

    # Export mean values
    f = open("Assortativity.txt", "w")

    for i in range(number_points):
        alfa = alfa_0 + i * step_alfa
        f.write("%f\t%f\t%f\n" % (alfa, assortativity[0][i], assortativity[1][i]))

    f.close()

    f = open("MCC.txt", "w")

    for i in range(number_points):
        alfa = alfa_0 + i * step_alfa
        f.write("%f\t%f\t%f\n" % (alfa, MCC[0][i], MCC[1][i]))

    f.close()

    f = open("GCC.txt", "w")

    for i in range(number_points):
        alfa = alfa_0 + i * step_alfa
        f.write("%f\t%f\t%f\n" % (alfa, GCC[0][i], GCC[1][i]))

    f.close()

    if show_plots:
        visualize = number_points // 2
        x = np.array([alfa_0 + i * step_alfa for i in range(number_points)])
        plt.errorbar(x, GCC[0], yerr=GCC[1], label="GCC")
        plt.errorbar(x, MCC[0], yerr=MCC[1], label="MCC")
        plt.errorbar(x, assortativity[0], yerr=assortativity[1], label="Assortativity")
        # plt.scatter(x,y)
        plt.ylabel("Mean value")
        # plt.yscale('log')
        plt.xlabel("alpha")
        plt.show()


################    main    ################################
N = 1000
# sizes=[50, 100, 250, 500]
iters = 10000
# n_max=25

alfa_0 = 0.3
number_points = 10
step_alfa = 0.4

# analyze_CCs(N, iters, alfa_0, number_points, step_alfa, n_max, compute_nCCs=False)
# analyze_VD(N, iters)
# analyze_micros(N, iters)

###  Algorithm complexity analysis.    #####
###  To start, set test_complex=True   #####

test_complex = False
if test_complex:
    puntos = 10
    start = np.zeros(puntos)
    finish = np.zeros(puntos)
    Nes = np.linspace(100, 1000, num=puntos)
    for i in range(puntos):
        N = int(Nes[i])
        print("N=", N)
        start[i] = time.time()
        analyze_CCs(N, iters)
        analyze_VD(N, iters)
        analyze_micros(N, iters)
        finish[i] = time.time()

    np.savetxt("N.txt", Nes)
    np.savetxt("times.txt", finish - start)

    plt.scatter(Nes, finish - start)
    plt.show()
