import random
import numpy as np
import matplotlib.pyplot as plt
from deap import base, creator, tools
from deap import algorithms
from functools import partial
from resource_allocation import fen

# Your fen function should be defined here

# Define the evaluation function
def evaluate(x1, x2):
    avg_sensor_delay, avg_sensor_speed, avg_video_speed, avg_video_delay = fen(x1, x2)
    return (avg_sensor_delay, avg_sensor_speed, avg_video_speed, avg_video_delay)

def fitness_function(individual):
    x1, x2 = individual
    result = evaluate(x1, x2)
    if result[0] < 5 and result[1] > 200 and result[2] > 200 and result[3] < 1000:
        return 1,
    else:
        return 0,

# Define the Genetic Algorithm
def genetic_algorithm():
    creator.create("FitnessMax", base.Fitness, weights=(1.0,))
    creator.create("Individual", list, fitness=creator.FitnessMax)

    toolbox = base.Toolbox()

    # Attribute generator
    toolbox.register("attr_int", random.randint, 0, 100)

    # Structure initializers
    toolbox.register("individual", tools.initRepeat, creator.Individual, toolbox.attr_int, 2)
    toolbox.register("population", tools.initRepeat, list, toolbox.individual)

    # Genetic operators
    toolbox.register("mate", tools.cxTwoPoint)
    toolbox.register("mutate", tools.mutUniformInt, low=0, up=100, indpb=0.2)
    toolbox.register("select", tools.selBest)
    toolbox.register("evaluate", fitness_function)

    pop = toolbox.population(n=100)
    hof = tools.HallOfFame(1)
    stats = tools.Statistics(lambda ind: ind.fitness.values)
    stats.register("Avg", np.mean)
    stats.register("Std", np.std)
    stats.register("Min", np.min)
    stats.register("Max", np.max)

    pop, logbook = algorithms.eaSimple(pop, toolbox, cxpb=0.5, mutpb=0.2, ngen=100, stats=stats, halloffame=hof, verbose=True)

    return hof[0]

# Run the Genetic Algorithm
best_solution = genetic_algorithm()
best_x1, best_x2 = best_solution
print(f"Best solution: x1={best_x1}, x2={best_x2}")

# Evaluate the best solution and plot the results
avg_sensor_delay, avg_sensor_speed, avg_video_speed, avg_video_delay = evaluate(best_x1, best_x2)

plt.figure()
plt.bar(["avg_sensor_delay", "avg_sensor_speed", "avg_video_speed", "avg_video_delay"],
        [avg_sensor_delay, avg_sensor_speed, avg_video_speed, avg_video_delay])
plt.ylabel("Values")
plt.title("Optimized Performance Metrics")
plt.show()