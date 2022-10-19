import os
import random as rd
import numpy as np
import math
import networkx as nx
import matplotlib.pyplot as plt
import collections
import time


def create_popul_Zanette(N, alpha):
    """
    Function that given a population size and growth-factor,
    returns the dictionaries:
    pc2cl ->  parent couple:  [list of children]
    c2p   ->  child: parent
    M     ->  size of parent's population (N/alpha)
    Each child could be assigned to any of the M couple of parents (rho=1.0).
    Note: No sex is assigned to individuals in this step
    """

    M = int(N / alpha)

    pc2cl = {i: [] for i in range(M)}
    c2p = {}

    for i in range(2 * N):
        p = rd.randint(0, M - 1)
        c2p[i] = p
        # print(i)
        pc2cl[p].append(i)

    return pc2cl, c2p, M


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


def generate_network_Zanette(N, alpha):
    """
    Generate a kinship network, according to D.H. Zanette's model.
    Input:
    - N : (int) size of population's males or females
    - alpha: (float) growth factor

    Output:
    - G: (Graph)
    -- nodes: men and women of the population
    -- edges: links between:
    ---         blood-siblings
    ---         spouses

    """
    ############# Create society ##################

    # Creating the F1 generation
    p2c, c2p, M = create_popul_Zanette(N, alpha)
    del M

    # Assignation of random marriages according to mating rules
    x = 0
    sucess, spouses_dict, gender = marriages_F1(N, p2c, c2p)

    while not sucess:
        x += 1
        sucess, spouses_dict, gender = marriages_F1(N, p2c, c2p)
        # print('Hizo falta volver a asignar matrimonios')
    del x

    ######################Build graph #################
    G = nx.Graph()
    popul_F1 = list(p2c.keys())
    G.add_nodes_from([i for i in range(2 * N)])  # Add nodes

    # Add blood siblings edges
    for e in list(p2c.keys()):
        L = p2c[e]

        for i in range(0, len(L) - 1):
            for j in range(i + 1, len(L)):
                G.add_edge(L[i], L[j])

    # Add marriage links
    gdr = "m"  # this is only for guidance
    for p in range(2 * N):
        if gender[p] == gdr:
            G.add_edge(p, spouses_dict[p])

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


### Experiments with Poissonian
def analyze_VD_Zanette(N, iters, alfa_0, number_points, step_alfa, show_plots=False):

    # Initialization of results containers
    vd_moments = np.zeros((2, number_points))
    vd_distr = []

    counter = 0

    print("###################  Starting VD   ###################")
    for i in range(number_points):

        # Update the value of 'alfa'
        alfa = alfa_0 + i * step_alfa

        ########################Create population##########################################

        k = np.array([])
        # Do this several times for
        for j in range(iters):
            counter += 1
            if 100 * counter % (iters * number_points) == 0:
                print("Program at %d  done" % (100 * counter / (iters * number_points)))

            # Generate the network
            G = generate_network_Zanette(N, alfa)

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


def analyze_CCs_Zanette(
    N, iters, alfa_0, number_points, step_alfa, show_plots=False, compute_nCCs=True
):

    # Initialization of results containers
    if compute_nCCs:
        CC_number = np.zeros(
            (2, number_points)
        )  # Container for mean value and standart deviation
        CC_number_rand = np.zeros((2, number_points))

    CC_sizes = []  # Conatiner for probability distribution functions
    CC_sizes_rand = []

    counter = 0

    print("###################  Starting CC   ###################")
    for i in range(number_points):

        # Update the value of 'alfa'
        alfa = alfa_0 + i * step_alfa

        ccsizes = np.array([])
        ccsizes_rand = np.array([])
        if compute_nCCs:
            ccnumber = np.array([])
            ccnumber_rand = np.array([])

        # Do this several times for
        for j in range(iters):
            counter += 1
            if 100 * counter % (iters * number_points) == 0:
                print("Program at %d  done" % (100 * counter / (iters * number_points)))

            # Generate the network
            G = generate_network_Zanette(N, alfa)
            r = randomize_net(G)

            # Analysis of graph
            if compute_nCCs:
                ncc = nx.algorithms.components.number_connected_components(G)
                ncc_rand = nx.algorithms.components.number_connected_components(r)

            scc = np.array([len(re) for re in nx.connected_components(G)])
            scc_rand = np.array([len(re) for re in nx.connected_components(r)])

            # Save the results
            if compute_nCCs:
                ccnumber = np.append(ccnumber, ncc)
                ccnumber_rand = np.append(ccnumber_rand, ncc_rand)
            ccsizes = np.append(ccsizes, scc)
            ccsizes_rand = np.append(ccsizes_rand, scc_rand)

        # Save the moments of the distribution
        if compute_nCCs:
            CC_number[0][i] = np.mean(ccnumber)
            CC_number[1][i] = np.std(ccnumber)
            CC_number_rand[0][i] = np.mean(ccnumber_rand)
            CC_number_rand[1][i] = np.std(ccnumber_rand)

        # Create a degree distribution for that value of 'a'
        c = collections.Counter(ccsizes)
        cr = collections.Counter(ccsizes_rand)
        # print('para <n>=',1/(a-1), 'componentes:', ccsizes)
        l_keys = np.array(list(c.keys()))
        l_keys = np.sort(l_keys)
        l_keys_r = np.array(list(cr.keys()))
        l_keys_r = np.sort(l_keys_r)

        # Normalization of counts
        tot = 0
        totr = 0
        for a in l_keys:
            tot += c[a]

        for a in l_keys_r:
            totr += cr[a]

        for t in l_keys:
            c[t] = c[t] / tot

        for t in l_keys_r:
            cr[t] = cr[t] / totr

        # Saving distribution
        CC_sizes.append(c)
        CC_sizes_rand.append(cr)
        seud_name = int(alfa * 1000)

        name = "CC_sizes_%d.txt" % seud_name
        name_r = "CC_sizes_%d_rand.txt" % seud_name

        documento = open(name, "w")
        for x in l_keys:
            # print((int(x), c[x]))
            documento.write("%d\t%f\n" % (int(x), c[x]))
        documento.close()

        documento = open(name_r, "w")
        for x in l_keys_r:
            # print((int(x), c[x]))
            documento.write("%d\t%f\n" % (int(x), cr[x]))
        documento.close()

    if compute_nCCs:
        # Export mean VD
        f = open("CC_moments.txt", "w")
        g = open("CC_moments_rand.txt", "w")

        for i in range(number_points):
            alfa = alfa_0 + i * step_alfa
            f.write("%f\t%f\t%f\n" % (alfa, CC_number[0][i], CC_number[1][i]))
            g.write("%f\t%f\t%f\n" % (alfa, CC_number_rand[0][i], CC_number_rand[1][i]))

        f.close()
        g.close()

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


def analyze_distances_Zanette(N, iters, alfa_0, number_points, step_alfa):

    # Initialization of results containers
    dist_moments = np.zeros((2, number_points))
    dist_moments_rand = np.zeros((2, number_points))

    counter = 0

    print("###################  Starting micros   ###################")
    for i in range(number_points):

        # Update the value of 'alfa'
        alfa = alfa_0 + i * step_alfa

        d = {a: 0 for a in range(2 * N)}
        d_rand = {a: 0 for a in range(2 * N)}

        # Do this several times for
        for j in range(iters):
            counter += 1
            if 100 * counter % (iters * number_points) == 0:
                print(
                    "Program at %d  done, alpha=%f"
                    % (100 * counter / (iters * number_points), alfa)
                )

            # Generate the network
            G = generate_network_Zanette(N, alfa)
            r = randomize_net(G)
            G0 = G.subgraph(
                sorted(nx.connected_components(G), key=len, reverse=True)[0]
            )
            r0 = r.subgraph(
                sorted(nx.connected_components(r), key=len, reverse=True)[0]
            )
            nodes = list(G0.nodes())
            nodes_rand = list(r0.nodes())

            for x in range(0, len(nodes) - 1):
                for y in range(x + 1, len(nodes)):
                    distance = len(
                        nx.shortest_path(G0, source=nodes[x], target=nodes[y])
                    )
                    d[distance] += 1

            for x in range(0, len(nodes_rand) - 1):
                for y in range(x + 1, len(nodes_rand)):
                    distance_rand = len(
                        nx.shortest_path(r0, source=nodes_rand[x], target=nodes_rand[y])
                    )
                    d_rand[distance_rand] += 1

        mean = 0
        counts = 0
        for h in range(2 * N):
            mean += d[h] * h
            counts += d[h]
        mean = mean / counts
        desv = 0
        for h in range(2 * N):
            desv += d[h] * (h - mean) ** 2
        desv = math.sqrt(desv / (counts - 1))

        dist_moments[0][i] = mean
        dist_moments[1][i] = desv

        ### Random mean
        mean = 0
        counts = 0
        for h in range(2 * N):
            mean += d_rand[h] * h
            counts += d_rand[h]
        mean = mean / counts
        desv = 0
        for h in range(2 * N):
            desv += d_rand[h] * (h - mean) ** 2
        desv = math.sqrt(desv / (counts - 1))

        dist_moments_rand[0][i] = mean
        dist_moments_rand[1][i] = desv

    # Export mean dist
    f = open("distance_moments.txt", "w")

    for i in range(number_points):
        alfa = alfa_0 + i * step_alfa
        f.write("%f\t%f\t%f\n" % (alfa, dist_moments[0][i], dist_moments[1][i]))

    f.close()
    # Export mean dist random
    f = open("distance_moments_random.txt", "w")

    for i in range(number_points):
        alfa = alfa_0 + i * step_alfa
        f.write(
            "%f\t%f\t%f\n" % (alfa, dist_moments_rand[0][i], dist_moments_rand[1][i])
        )

    f.close()


def analyze_distances_N(sizes, iters, alpha=1.0):
    number_points = len(sizes)
    # Initialization of results containers
    dist_moments = np.zeros((2, len(sizes)))
    dist_moments_rand = np.zeros((2, number_points))

    counter = 0

    print("###################  Starting micros   ###################")
    for i in range(number_points):

        # Update the value of 'alfa'
        # alfa=alfa_0+i*step_alfa
        alfa = alpha
        N = sizes[i]

        d = {a: 0 for a in range(2 * N)}
        d_rand = {a: 0 for a in range(2 * N)}

        # Do this several times for
        for j in range(iters):
            counter += 1
            if 100 * counter % (iters * number_points) == 0:
                print(
                    "Program at %d  done, N=%f"
                    % (100 * counter / (iters * number_points), N)
                )

            # Generate the network
            G = generate_network_Zanette(N, alfa)
            r = randomize_net(G)
            G0 = G.subgraph(
                sorted(nx.connected_components(G), key=len, reverse=True)[0]
            )
            r0 = r.subgraph(
                sorted(nx.connected_components(r), key=len, reverse=True)[0]
            )
            nodes = list(G0.nodes())
            nodes_rand = list(r0.nodes())

            for x in range(0, len(nodes) - 1):
                for y in range(x + 1, len(nodes)):
                    distance = len(
                        nx.shortest_path(G0, source=nodes[x], target=nodes[y])
                    )
                    d[distance] += 1

            for x in range(0, len(nodes_rand) - 1):
                for y in range(x + 1, len(nodes_rand)):
                    distance_rand = len(
                        nx.shortest_path(r0, source=nodes_rand[x], target=nodes_rand[y])
                    )
                    d_rand[distance_rand] += 1

        mean = 0
        counts = 0
        for h in range(2 * N):
            mean += d[h] * h
            counts += d[h]
        mean = mean / counts
        desv = 0
        for h in range(2 * N):
            desv += d[h] * (h - mean) ** 2
        desv = math.sqrt(desv / (counts - 1))

        dist_moments[0][i] = mean
        dist_moments[1][i] = desv

        ### Random mean
        mean = 0
        counts = 0
        for h in range(2 * N):
            mean += d_rand[h] * h
            counts += d_rand[h]
        mean = mean / counts
        desv = 0
        for h in range(2 * N):
            desv += d_rand[h] * (h - mean) ** 2
        desv = math.sqrt(desv / (counts - 1))

        dist_moments_rand[0][i] = mean
        dist_moments_rand[1][i] = desv

    # Export mean dist
    f = open("distance_moments_N.txt", "w")

    for i in range(number_points):
        N = sizes[i]
        f.write("%f\t%f\t%f\n" % (N, dist_moments[0][i], dist_moments[1][i]))

    f.close()
    # Export mean dist random
    f = open("distance_moments_N_random.txt", "w")

    for i in range(number_points):
        N = sizes[i]
        f.write("%f\t%f\t%f\n" % (N, dist_moments_rand[0][i], dist_moments_rand[1][i]))

    f.close()


################    main    ################################
N = 1000
# sizes=[50, 100, 250, 500]
iters = 10000
# n_max=25

alfa_0 = 0.3
number_points = 10
step_alfa = 0.4

# analyze_VD_Zanette(N, iters, alfa_0, number_points, step_alfa, show_plots=True)
analyze_CCs_Zanette(
    N, iters, alfa_0, number_points, step_alfa, show_plots=False, compute_nCCs=False
)
# analyze_distances_Zanette(N, iters, alfa_0, number_points, step_alfa)
# analyze_distances_N(sizes, iters, alpha=1.0)

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
        ##  FUNCTIONS OR EXPERIMENTS TO RUN
    np.savetxt("times.txt", finish - start)

    plt.scatter(Nes, finish - start)
    plt.show()
