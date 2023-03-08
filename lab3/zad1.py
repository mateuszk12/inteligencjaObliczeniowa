import pygad
import math

#definiujemy parametry chromosomu
#geny to liczby: 0 lub 1
gene_space = [round(i*0.01,2) for i in range(0,100,1)]
def endurance(x):
    return math.exp(-2*(x[1]-math.sin(x[0]))**2)+math.sin(x[2]*x[3])+math.cos(x[4]*x[5])

def fitness_func(solution, solution_idx):
    fitness = endurance(solution)
    return fitness

fitness_function = fitness_func


sol_per_pop = 1000
num_genes = 6


num_parents_mating = 5
num_generations = 30
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
                       mutation_percent_genes=mutation_percent_genes)


ga_instance.run()

solution = ga_instance.best_solution()
print(solution)

