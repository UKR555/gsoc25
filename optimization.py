import random
import numpy as np
import matplotlib.pyplot as plt
from deap import base, creator, tools, algorithms

# Define multi-objective optimization: Maximize utility, Minimize risk
creator.create("FitnessMulti", base.Fitness, weights=(1.0, -1.0))  # Max utility, Min risk
creator.create("Individual", list, fitness=creator.FitnessMulti)

def evaluate(individual):
    utility = individual[0]  # Example: Utility function
    risk = individual[1]  # Example: Risk function (to be minimized)
    return utility, risk

toolbox = base.Toolbox()
toolbox.register("attr_float", random.uniform, 0, 10)  # Random values between 0 and 10
toolbox.register("individual", tools.initRepeat, creator.Individual, toolbox.attr_float, n=2)
toolbox.register("population", tools.initRepeat, list, toolbox.individual)

toolbox.register("mate", tools.cxBlend, alpha=0.5)
toolbox.register("mutate", tools.mutGaussian, mu=0, sigma=1, indpb=0.2)
toolbox.register("select", tools.selNSGA2)
toolbox.register("evaluate", evaluate)

def main():
    population = toolbox.population(n=100)
    hof = tools.HallOfFame(10)  
    stats = tools.Statistics(lambda ind: ind.fitness.values)
    stats.register("avg", np.mean, axis=0)
    
    algorithms.eaMuPlusLambda(population, toolbox, mu=100, lambda_=200, 
                              cxpb=0.7, mutpb=0.2, ngen=50, 
                              stats=stats, halloffame=hof, verbose=True)

    return population, hof

if __name__ == "__main__":
    population, hof = main()
    pareto_front = np.array([ind.fitness.values for ind in hof])

    # Plot Pareto Front
    plt.scatter(pareto_front[:, 0], pareto_front[:, 1], c="red", label="Pareto Front")
    plt.xlabel("Utility (Maximize)")
    plt.ylabel("Risk (Minimize)")
    plt.title("Pareto Front of Multi-Objective Optimization")
    plt.legend()
    plt.grid()
    plt.show()
