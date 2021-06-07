from random import randrange


class Route:
    """
    Represents route over cities.
    """

    def __init__(self, route):
        """
        @arg route is a list of City objects.
        """
        self.route = route
        self.length = len(route)
        self.route_distance = None
 
                     
    def calculate_distance(self, tsp):
        self.route_distance = self.distance(tsp)


    def distance(self, tsp):
        """
        Calculates distance over route for given TSP instance.
        """
        route_distance = 0
        for i in range(1,len(self.route)):
           route_distance += tsp.distance_matrix(self.route[i-1].city_number, self.route[i].city_number)
        return route_distance
    

    def two_opt(self): 
        """
        Swaps two random different cities (two_opt method).
        """
        assert len(self.route) >= 2

        index_1 = randrange(len(self.route))
        index_2 = randrange(len(self.route))
        while index_1 == index_2:
            index_2 = randrange(len(self.route))
        self.route[index_1], self.route[index_2] = self.route[index_2], self.route[index_1]

    
    def copy(self):
        new_route = Route(self.route.copy())
        # print(id(new_route))
        return new_route

