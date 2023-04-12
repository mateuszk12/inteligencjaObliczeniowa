import pygad
import networkx as nx
import matplotlib.pyplot as plt


graph = [(1, 2), (2, 3), (3, 4), (4, 5), (5, 6), (6, 7), (7, 8), (8, 9), (9, 10), (10, 11), (11, 12), (12, 13), (13, 14), (14, 1), (1, 5), (5, 9), (9, 13), (13, 2), (2, 6), (6, 10), (10, 14), (14, 3), (3, 7), (7, 11), (11, 4), (4, 8), (8, 12), (12, 1)]
def hamilton1(graph,results:bool):
    flatten = [item for sub_list in graph for item in sub_list]
    number_of_nodes = len(set(flatten))
    gene_space = [i for i in range(len(graph))]


    def fitness_function(solution, solution_idx):
        fitness = 0
        path = [graph[int(i)] for i in solution]
        visited = {}
        flatten = [item for sub_list in path for item in sub_list]
        incoming = {}
        outcoming = {}
        repeat_counter = 0
        repeat_incoming = 0
        repeat_outcoming = 0
        # ile i które wierzchołki zostały odwiedzone
        for i in path:
            if i[0] not in incoming:
                incoming[i[0]] = 1
            else:
                repeat_incoming += 1
                incoming[i[0]] += 1
            if i[1] not in outcoming:
                outcoming[i[1]] = 1
            else:
                repeat_outcoming += 1
                outcoming[i[1]] += 1
        nodes_incoming = len(incoming)
        nodes_outcoming = len(outcoming)
        fitness -= 2000 * (number_of_nodes - nodes_incoming) * number_of_nodes
        fitness -= 2000 * (number_of_nodes - nodes_outcoming) * number_of_nodes
        fitness -= 2000 * (repeat_outcoming) * number_of_nodes
        fitness -= 2000 * (repeat_incoming) * number_of_nodes
        for i in visited:
            if visited[i] != 2:
                fitness -= 1000 * number_of_nodes * visited[i]
        if repeat_counter > 0:
            fitness -= 3000 * number_of_nodes * repeat_counter - 1

        # sprawdzanie spójności grafu
        for i in range(len(path)):
            if i > 0:
                if path[i][0] == path[i - 1][1]:
                    fitness += 900 * number_of_nodes
                else:
                    fitness -= 1200 * number_of_nodes

        # sprawdza czy to cykl
        if path[0][0] == path[-1][1]:
            fitness += 1000 * number_of_nodes
        else:
            fitness -= 3000 * number_of_nodes

        return fitness


    sol_per_pop = 10 * number_of_nodes
    num_genes = number_of_nodes
    num_parents_mating = 5
    num_generations = 5 * number_of_nodes
    keep_parents = 2
    parent_selection_type = "sss"
    crossover_type = "scattered"
    mutation_type = "random"
    mutation_percent_genes = 18
    ga_instance = pygad.GA(gene_space=gene_space,
                        num_generations=num_generations,
                        num_parents_mating=num_parents_mating,
                        fitness_func=fitness_function,
                        sol_per_pop=sol_per_pop,
                        num_genes=num_genes,
                        parent_selection_type=parent_selection_type,
                        keep_parents=keep_parents,
                        crossover_type=crossover_type,
                        mutation_type=mutation_type,
                        mutation_percent_genes=mutation_percent_genes,
                        stop_criteria=[f"saturate_{60}"],
                        allow_duplicate_genes=False)

    ga_instance.run()
    if results:
        ga_instance.plot_fitness()
    solution = ga_instance.best_solution()
    optimal_nodes = [graph[int(i)] for i in solution[0]]

    flatten = [item for sub_list in optimal_nodes for item in sub_list]
    connected = True
    wrong_edges = []
    for i in range(len(optimal_nodes)):
            if i > 0:
                if optimal_nodes[i][0] != optimal_nodes[i - 1][1]:
                    connected = False
                    wrong_edges.append(optimal_nodes[i])
    if optimal_nodes[0][0] != optimal_nodes[-1][1]:
        connected = False
        wrong_edges.append(optimal_nodes[0])
    print(wrong_edges)
    if results:
        print(optimal_nodes)
        G = nx.DiGraph(directed=True)
        for i in flatten:
            G.add_node(i)
        for i in graph:
            G.add_edge(*i)
        options = {
            "font_size": 17,
            "node_size": 500,
            "node_color": "white",
            "edgecolors": "black",
            "linewidths": 3,
            "width": 1,
            'arrowstyle': '-|>',
            'arrowsize': 7,
        }
        for i in G.edges():
            if i in optimal_nodes[0:number_of_nodes] and i not in wrong_edges:
                G[i[0]][i[1]]['color'] = 'green'
            elif i in wrong_edges:
                G[i[0]][i[1]]['color'] = 'red'
            else:
                G[i[0]][i[1]]['color'] = 'black'
        edge_color_list = [G[e[0]][e[1]]['color'] for e in G.edges()]
        black = [edge for edge in G.edges() if edge not in optimal_nodes]
        pos = nx.circular_layout(G)
        nx.draw_networkx(G, pos, arrows=True, edge_color=edge_color_list, connectionstyle="arc3,rad=0.2", **options)
        ax = plt.gca()
        ax.margins(0.20)
        plt.axis("off")
        plt.show()
    
    if connected:
        if results:
            (print("jest cykl"))
        return True
    else:
        if results:
            (print("brak cyklu"))
        return False
hamilton1(graph,True)