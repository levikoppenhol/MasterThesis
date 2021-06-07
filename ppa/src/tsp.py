import numpy as np


from city import City


class TSP:
    """
    Represents Traveling Salesman Problem instance.
    """

    def __init__(self, path):
        self.dimension, self.cities = TSP.read_tsp_problem(path)
        self.matrix = self.route_to_matrix()


    def route_to_matrix(self):
        route_matrix = np.zeros((int(self.dimension), int(self.dimension)))  
        for x in range(self.dimension):     
            for y in range(x, self.dimension):
                route_matrix[x, y] = self.cities[x].distance(self.cities[y])
                route_matrix[y, x] = route_matrix[x,y]
        return route_matrix


    def distance_matrix(self, city1, city2):
        return self.matrix[city1-1, city2-1]


    @staticmethod
    def read_numbered_city_line(words, solution):
        assert len(words) >= 3
        city_number = int(words[0])
        x, y = None, None
        if solution == False:
            x = float(words[1])
            y = float(words[2])
        return City(city_number, x, y)


    @staticmethod
    def read_cities_problem(tspfile):
        # Read dimension.
        dimension = None
        for line in tspfile:
            if 'DIMENSION' in line:
                dimension = int(line.split()[-1])
            elif 'TOUR_SECTION' in line or 'NODE_COORD_SECTION' in line:
                assert dimension is not None
                break 
        assert dimension is not None

        # Read cities.
        cities = []  
        for _ in range(dimension):
            line  = tspfile.readline()
            words = line.split()
            cities.append(TSP.read_numbered_city_line(words, False))
        return dimension, cities
        

    @staticmethod
    def read_tsp_problem(path):
        with open(path,'r') as tspfile:
            dimension, cities = TSP.read_cities_problem(tspfile)
            return dimension, cities
        assert False
