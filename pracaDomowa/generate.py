import random
def generate(num,nodes):
    graph = []
    nodes = nodes % 26
    number_of_nodes = 0
    while len(graph) != num and number_of_nodes != nodes:
        flatten = [item for sub_list in graph for item in sub_list]
        number_of_nodes = len(set(flatten))
        if len(graph) > 0:
            node1 = graph[-1][1]
            node2 = chr((int(random.random() * 100) % nodes) + 65)
        else:
            node1 = chr((int(random.random()*100) % nodes) + 65)
            node2 = chr((int(random.random() * 100) % nodes) + 65)
        edge = (node1,node2)
        if edge not in graph and node1 != node2:
            if len(edge) == num-1:
                node3 = graph[0][0]
                graph.append((node1,node3))
            else:
                graph.append(edge)
    return graph
