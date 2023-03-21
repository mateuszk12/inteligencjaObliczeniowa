import pygad
import numpy
import time
labirynt = [[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 2, 1, 1, 0, 1, 1, 1, 0, 1, 1, 0],
            [0, 0, 0, 1, 1, 1, 0, 1, 0, 0, 1, 0],
            [0, 1, 1, 1, 0, 1, 0, 1, 1, 1, 1, 0],
            [0, 1, 0, 1, 0, 0, 1, 1, 0, 0, 1, 0],
            [0, 1, 1, 0, 0, 1, 1, 1, 0, 1, 1, 0],
            [0, 1, 1, 1, 1, 1, 0, 1, 1, 1, 0, 0],
            [0, 1, 0, 1, 1, 0, 0, 1, 0, 1, 1, 0],
            [0, 1, 0, 0, 0, 1, 1, 1, 0, 0, 1, 0],
            [0, 1, 0, 1, 0, 0, 1, 0, 1, 0, 1, 0],
            [0, 1, 0, 1, 1, 1, 1, 1, 1, 1, 3, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]]

gene_space = [0, 1, 2, 3]
# 0 lewo
# 1 gora
# 2 prawo
# 3 dol
# jeżeli t
def move(i, x, y):
    match i:
        case 0:
            return [x - 1, y]
        case 1:
            return [x, y - 1]
        case 2:
            return [x + 1, y]
        case 3:
            return [x, y + 1]


def fitness_func(solution, solution_idx):
    def fitness(x,y,steps,hits):
        endy = abs(x - 10)
        endx = abs(y - 10)
        distance = endy + endx
        return (steps*100 - hits*301 - distance*60)
    good = 0
    hit = 0
    x = 1
    y = 1
    for i in solution:
        tmpx = move(i, x, y)[0]
        tmpy = move(i, x, y)[1]
        if labirynt[tmpy][tmpx] == 0:
            hit += 1
        else:
            good += 1
            x = tmpx
            y = tmpy
            if labirynt[y][x] == 3:
                good += 5000
                return fitness(x,y,good,hit)
        if good > 30:
            break
    return fitness(x, y, good, hit)


fitness_function = fitness_func

# ile chromsomów w populacji
# ile genow ma chromosom
sol_per_pop = 100
num_genes = 30

# ile wylaniamy rodzicow do "rozmanazania" (okolo 50% populacji)
# ile pokolen
# ilu rodzicow zachowac (kilka procent)
num_parents_mating = 100
num_generations = 1000
keep_parents = 8

# jaki typ selekcji rodzicow?
# sss = steady, rws=roulette, rank = rankingowa, tournament = turniejowa
parent_selection_type = "sss"

# w il =u punktach robic krzyzowanie?
crossover_type = "single_point"

# mutacja ma dzialac na ilu procent genow?
# trzeba pamietac ile genow ma chromosom
mutation_type = "random"
mutation_percent_genes = 8

# inicjacja algorytmu z powyzszymi parametrami wpisanymi w atrybuty
for i in range(10):
    start=time.time()
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
                           )
    ga_instance.run()
    end=time.time()
    print('czas: ',end-start)
    solution, solution_fitness, solution_idx = ga_instance.best_solution()
    print("Parameters of the best solution : {solution}".format(solution=solution))
    print("Fitness value of the best solution = {solution_fitness}".format(solution_fitness=solution_fitness))
# podsumowanie: najlepsze znalezione rozwiazanie (chromosom+ocena)

