import pygad
import networkx as nx
import matplotlib.pyplot as plt
from generate import generate

graph = [(1, 2), (2, 3), (3, 4), (4, 5), (5, 6), (6, 7), (7, 8), (8, 9), (9, 10), (10, 11), (11, 12), (12, 13), (13, 14), (14, 1), (1, 5), (5, 9), (9, 13), (13, 2), (2, 6), (6, 10), (10, 14), (14, 3), (3, 7), (7, 11), (11, 4), (4, 8), (8, 12), (12, 1)]


flatten = [item for sub_list in graph for item in sub_list]
number_of_nodes = len(set(flatten))
nodes = list(set(flatten))
gene_space = [i for i in range(number_of_nodes)]


def fitness_function(solution, solution_idx):
    fitness = 0
    first_node = solution[0]
    #czy krawęz istnieje
    for i in range(len(solution)-1):
        node1 = nodes[int(solution[i])]
        node2 = nodes[int(solution[i+1])]
        if (node1,node2) in graph:
            fitness += 400*len(solution)
        else:
            fitness -= 900*len(solution)
    #czy sie domyka
    if (nodes[int(solution[-1])], nodes[int(solution[0])]) in graph:
        fitness += 400 * len(solution)
    else:
        fitness -= 900 * len(solution)
    return fitness

sol_per_pop = 100 * number_of_nodes
num_genes = number_of_nodes
num_parents_mating = 5
num_generations = 5 * number_of_nodes
keep_parents = 2
parent_selection_type = "sss"
crossover_type = "single_point"
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
ga_instance.plot_fitness()
solution = ga_instance.best_solution()
optimal_nodes = [nodes[int(i)] for i in solution[0]]
if (optimal_nodes[-1],optimal_nodes[0]) in graph:
    optimal_nodes.append(optimal_nodes[0])

flatten = []
print(optimal_nodes)
for i in range(len(optimal_nodes)-1):
    flatten.append((optimal_nodes[i],optimal_nodes[i+1]))



G = nx.DiGraph(directed=True)
for i in optimal_nodes:
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
    if i in flatten[0:number_of_nodes]:
        G[i[0]][i[1]]['color'] = 'red'
    else:
        G[i[0]][i[1]]['color'] = 'black'
edge_color_list = [G[e[0]][e[1]]['color'] for e in G.edges()]
black = [edge for edge in G.edges() if edge not in optimal_nodes]
pos = nx.circular_layout(G)
nx.draw_networkx(G,pos ,arrows=True, edge_color=edge_color_list, connectionstyle="arc3,rad=0.2", **options)
ax = plt.gca()
ax.margins(0.20)
plt.axis("off")
plt.show()
