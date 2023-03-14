import pygad
import numpy
labirynt = [
             [2, 1, 1, 0, 1, 1, 1, 0, 1, 1 ],
             [0, 0, 1, 1, 1, 0, 1, 0, 0, 1 ],
             [1, 1, 1, 0, 1, 0, 1, 1, 1, 1 ],
             [1, 0, 1, 0, 0, 1, 1, 0, 0, 1 ],
             [1, 1, 0, 0, 1, 1, 1, 0, 1, 1 ],
             [1, 1, 1, 1, 1, 0, 1, 1, 1, 0 ],
             [1, 0, 1, 1, 0, 0, 1, 0, 1, 1 ],
             [1, 0, 0, 0, 1, 1, 1, 0, 0, 1 ],
             [1, 0, 1, 0, 0, 1, 0, 1, 0, 1 ],
             [1, 0, 1, 1, 1, 1, 1, 1, 1, 3]]
#1 - 100 + (-1*(abs(x - 9)+abs(y - 9)))
#2 - 50
gene_space = [0,1,2,3]
#0 lewo
#1 gora
#2 prawo
#3 dol
#jeżeli t
def move(i,x,y):
    match i:
        case 0:
            return [x - 1,y]
        case 1:
            return [x,y - 1]
        case 2:
            return [x + 1,y]
        case 3:
            return [x,y + 1]
def fitness_func(solution, solution_idx):
    fitness = 0
    good = 0
    x = 0
    y = 0
    for i in solution:
        hit = 0
        tmpx = move(i,x,y)[0]
        tmpy = move(i,x,y)[1]
        if tmpx in range(0,10) and tmpy in range(0,10):
            if labirynt[tmpx][tmpy] in range(1,4):
                x = tmpx
                y = tmpy
                good = 100 * (x+1) * (y+1)
                if labirynt[x][y] == 3:
                    return good
            elif labirynt[tmpx][tmpy] == 3:
                return good
            else:
                hit -= 1000
        else:
            hit -= 1000
        good -= hit
    return good




fitness_function = fitness_func

#ile chromsomów w populacji
#ile genow ma chromosom
sol_per_pop = 2000
num_genes = 30

#ile wylaniamy rodzicow do "rozmanazania" (okolo 50% populacji)
#ile pokolen
#ilu rodzicow zachowac (kilka procent)
num_parents_mating = 1000
num_generations = 100
keep_parents = 8

#jaki typ selekcji rodzicow?
#sss = steady, rws=roulette, rank = rankingowa, tournament = turniejowa
parent_selection_type = "sss"

#w il =u punktach robic krzyzowanie?
crossover_type = "single_point"

#mutacja ma dzialac na ilu procent genow?
#trzeba pamietac ile genow ma chromosom
mutation_type = "random"
mutation_percent_genes = 5

#inicjacja algorytmu z powyzszymi parametrami wpisanymi w atrybuty
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
                       mutation_percent_genes=mutation_percent_genes)

#uruchomienie algorytmu
ga_instance.run()

#podsumowanie: najlepsze znalezione rozwiazanie (chromosom+ocena)
solution, solution_fitness, solution_idx = ga_instance.best_solution()
print("Parameters of the best solution : {solution}".format(solution=solution))
print("Fitness value of the best solution = {solution_fitness}".format(solution_fitness=solution_fitness))



