import numpy as np
import matplotlib.pyplot as plt

# Objective function
def objective_function(position):
    return np.sum(position**2)

# Whale Optimization Algorithm (WOA)
class WOA:
    def __init__(self, population_size, search_space, num_iterations, b, a):
        self.population_size = population_size
        self.search_space = search_space
        self.num_iterations = num_iterations
        self.b = b
        self.a = a

        self.population = np.random.rand(population_size, 2) * (search_space[:, 1] - search_space[:, 0]) + search_space[:, 0]

    def run(self):
        global_best = np.min([objective_function(whale) for whale in self.population])
        global_best_position = self.population[np.argmin([objective_function(whale) for whale in self.population])]

        for iteration in range(self.num_iterations):
            self.a -= (1 / self.num_iterations)  # Linearly decreasing a

            for i, whale in enumerate(self.population):
                r1 = np.random.rand()
                r2 = np.random.rand()
                A = 2 * self.a * r1 - self.a
                C = 2 * r2

                p = np.random.rand()
                b_rand = 1  # Random value for b

                if p < 0.5:
                    if abs(A) < 1:
                        D = abs(C * global_best_position - whale)
                        self.population[i] = global_best_position - A * D
                    elif abs(A) >= 1:
                        rand_leader = self.population[np.random.randint(len(self.population))]
                        D = abs(C * rand_leader - whale)
                        self.population[i] = rand_leader - A * D
                else:
                    D = abs(global_best_position - whale)
                    self.population[i] = D * np.exp(self.b * b_rand) * np.cos(2 * np.pi * b_rand) + global_best_position

                # Update global best
                if objective_function(whale) < global_best:
                    global_best = objective_function(whale)
                    global_best_position = whale

        return global_best, global_best_position

def main():
    # Parameters
    num_whales = 30
    num_iterations = 50
    search_space = np.array([[-5, 5], [-5, 5]])
    b = 1
    a = 2

    # Initialize WOA
    woa_instance = WOA(num_whales, search_space, num_iterations, b, a)

    # Run WOA
    global_best, global_best_position = woa_instance.run()

    # Print results
    print("Global Best:", global_best)
    print("Global Best Position:", global_best_position)

    # Plot result
    # Plot result
    fig, ax = plt.subplots()
    ax.scatter(woa_instance.population[:, 0], woa_instance.population[:, 1], c='blue', label='Whales')
    ax.scatter(global_best_position[0], global_best_position[1], c='red', marker='X', label='Global Best')
    ax.set_xlim(search_space[0, 0], search_space[0, 1])
    ax.set_ylim(search_space[1, 0], search_space[1, 1])
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.legend()
    plt.show()

if __name__ == "__main__":
    main()