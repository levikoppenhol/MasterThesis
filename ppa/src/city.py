import numpy as np


class City:
    """
    Represents city. test
    """

    def __init__(self, city_number = None, x = None, y = None):
        self.city_number = city_number
        self.x = x
        self.y = y


    def distance(self, other):
        """
        Calculates Euclidean distance.
        """
        assert isinstance(other, City)
        assert self.x is not None 
        assert self.y is not None 
        assert other.x is not None 
        assert other.y is not None

        return int(np.sqrt((self.x - other.x)**2 + (self.y - other.y)**2))
  

    def __repr__(self):  
        return f"{self.city_number}"
        

    def __str__(self):
        return f"{self.city_number} {self.x} {self.y}"
        

