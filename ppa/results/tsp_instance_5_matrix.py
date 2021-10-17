import numpy as np 

# dimensions 30x24 
tsp_instance_5_1_city_list = [(1, 0.0, 13.0), (2, 10.0, 7.0), (3, 8.0, 5.0), (4, 18.0, 24.0), (5, 13.0, 0.0)] 
tsp_instance_5_1 = np.matrix([
[np.inf ,11 ,11 ,21 ,18 ],
[11 ,np.inf ,2 ,18 ,7 ],
[11 ,2 ,np.inf ,21 ,7 ],
[21 ,18 ,21 ,np.inf ,24 ],
[18 ,7 ,7 ,24 ,np.inf ]
])
