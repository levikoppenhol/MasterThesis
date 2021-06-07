import copy
from numpy import around
import random
from random import randrange
import time
import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt

from route import Route
from tsp import TSP 


class PPA:


    def __init__(self, evaluations, pop_size, n_max, s_max, instance):
        self.instance = instance
        self.evaluations = evaluations
        self.number_evaluations = 0 
        self.pop_size = pop_size
        self.n_max = n_max
        self.s_max = s_max
        self.route_matrix = instance.matrix
        self.population = self.initialize_population(instance, pop_size)
        self.population_offspring = []
        self.generations = 0
        self.x_max = 0
        self.x_min = np.inf 
        self.end_generation = True
        self.best_route = None
        
    
    def initialize_population(self, tsp_instance, pop_size):
        route = Route(tsp_instance.cities)
        initial_pop = [Route(random.sample(route.route, route.length)) for n in range(self.pop_size)]
        for route in initial_pop:
            route.calculate_distance(tsp_instance)
            self.number_evaluations += 1
         
        return initial_pop        
    

    def fitness(self, route):
        """
        Normalize and calculate the fitness,
        f(xi) within an interval between [0,1] with 
        z_x_i = (f(x_max)-f(x_i))/(f(x_max)-f(x_min)) 
        """
        normalized_fitness = (self.x_max - route.route_distance) / (self.x_max - self.x_min)
        F_xi = 0.5 * (np.tanh(4 * (normalized_fitness) - 2) + 1)
        return F_xi
    

    def assign_offspring(self, F_xi):
        return int(np.ceil(self.n_max * F_xi * random.random()))
    

    def mutation(self, F_xi):
        return int(np.ceil(self.s_max * random.random() * (1 - F_xi)))    
    

    def create_new_offspring(self, route, number_offspring, number_mutation):
        for _ in range(1, number_offspring + 1):   
            offspring_route = route.copy()
            assert id(offspring_route) != id(route)

            for _ in range(1, number_mutation+1):
                offspring_route.two_opt()

            offspring_route.calculate_distance(self.instance)
            self.population_offspring.append(offspring_route)
            self.number_evaluations += 1

     
    def evaluate(self):
        """
        Evaluate while that there wont be more evaluations that 
        specified but if the eveluation is busy it will finish it.
        """
        while self.end_generation:
           
            sorted_route_list = sorted([x.route_distance for x in self.population])  
            self.x_max = sorted_route_list[-1]
            self.x_min = sorted_route_list[0]

            for route in self.population:
                if  self.number_evaluations >= self.evaluations:
                    self.end_generation = False
                
                # The fitness is calculated for eacht route. 
                # If the X_max is equal to the X_min than all 
                # the fitness values will be 0.5
                if(self.x_max == self.x_min):
                    F_xi = 0.5
                else: 
                    F_xi = self.fitness(route)
                
                number_offspring = self.assign_offspring(F_xi)
                number_mutation = self.mutation(F_xi)     
                self.create_new_offspring(route, number_offspring, number_mutation)
                
        
            self.population = self.population + self.population_offspring
            self.population = sorted(self.population, key=lambda route: route.route_distance)[:self.pop_size]
        
            self.population_offspring = []

            # Save the best route and update if there is a better route.
            if self.best_route is None:
                self.best_route = self.population[0]
            elif self.best_route.route_distance > self.population[0].route_distance:
                self.best_route = self.population[0]
               
        return self.best_route.route_distance


def main():
    tsp = TSP("../data/instance_1.tsp.txt")     
    n_pop_size = 40 
    n_offspring = 10
    n_evaluations = 5000
    n_mutations = 20
    n_runs = 5

    arr = np.zeros((n_pop_size -1, n_offspring -1))

    
    for pop_size_i, pop_size in enumerate(range(2, n_pop_size + 1)):
        for offspring_i, max_offspring in enumerate(range(2, n_offspring + 1)):
            
            start = time.time()

            value = 0 
            
            for _ in range(n_runs):
                ppa = PPA(n_evaluations, pop_size, max_offspring, n_mutations, tsp)
                value += ppa.evaluate()
            mean = float(value) / n_runs

            end = time.time()

            print(f"Processing time: {round(end - start, 3)} seconds")
            arr[pop_size_i, offspring_i] = mean
            print(f"number of pop_size: {pop_size} and countsize for pop,off {pop_size_i},{offspring_i} number of offspring {max_offspring} with {mean} value")
    
   
    fig = plt.figure(figsize=(18, 9.6))
    ax = fig.add_subplot(111)
    ax.set_title('colorMap')
    plt.imshow(arr)
    ax.set_aspect('equal')

    plt.colorbar(orientation='vertical')
    plt.show()


if __name__ == '__main__':
    main()
    

