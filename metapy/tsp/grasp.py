'''
Greedy Randomised Adaptive Search Procedure

classes and functions.
'''

import numpy as np

class FixedRCLSizer:
    '''
    Fixed sized RCL list.
    
    When r = 1 then greedy
    When r = len(tour) then random
    '''
    def __init__(self, r):
        self.r = r
        
    def get_size(self):
        '''
        Returns an int representing the size of the required RCL
        '''
        return self.r
    
class RandomRCLSizer:
    '''
    Probabilitic selection of the RCL size
    Uniform probability.
    '''
    def __init__(self, r_list, random_seed=None):
        self.r_list = r_list
        self.rng = np.random.default_rng(random_seed)
        
    def get_size(self, size=None):
        '''
        Returns a randomly selected RCL size
        '''
        return self.rng.choice(self.r_list, size=size)
    
class SemiGreedyConstructor:
    '''
    Semi-greedy construction of a tour.
    
    For a city i creates a restricted candidate list of size r
    i.e the r shortest distances from city i.  
    Next city is chosen with equal probability.
    Repeats until tour is constructed.
    '''
    def __init__(self, rcl_sizer, tour, matrix,
                 random_seed=None):
        '''
        Constructor
        
        Params:
        ------
        rcl_sizer: object
            sizes the restricted candidate list
        
        tour: np.ndarray
            vector of city indexes included in problem
            
        matrix: np.ndarray
            matrix of travel costs
            
        random_seed: int
            used to control sampling and provides a
            reproducible result.
        '''
        
        # size of rcl
        self.rcl_sizer = rcl_sizer
        
        # cities in a tour
        self.tour = tour
        
        # travel cost matrix
        self.matrix = matrix
        
        # create random number generator
        self.rng = np.random.default_rng(random_seed)
    
    def build(self):
        '''
        Semi-greedy contruction of tour
        
        Returns:
        --------
        np.array
        '''
        # first city in tour
        solution = np.array([self.tour[0]])    
        
        # it is an iterative (construction) procedure
        for i in range(len(self.tour)-1):
            # get the RCL size
            r = self.rcl_sizer.get_size()
            
            # get the RCL
            rcl = self.get_rcl(r, solution, solution[-1])
            
            # select the next city 
            next_city = self.random_from_rcl(rcl)
            
            # update the solution
            solution = np.append(solution, np.array([next_city]))
            
        return solution
    
    def get_rcl(self, r, solution, from_city):
        '''
        Restricted candidate list for final city in current solution
        
        Params:
        -------
        solution: np.ndarray
            vector of current partially constructed solution
            
        from_city: int
            index of city used to construct rcl.
        
        Returns:
        -------
        np.array
        '''
        # get indexes of cities not in solution
        mask = self.tour[~np.in1d(self.tour, solution)]
        
        # get indexes of r smallest travels costs 
        if mask.shape[0] > r:
            # partition the vector for remaining cities - faster than sorting 
            idx = np.argpartition(self.matrix[from_city][mask], 
                                  len(mask) - r)[-r:]
            rcl = mask[idx]
        else:
            # handle when r < n cities remaining 
            rcl = mask
        return rcl
    
    def random_from_rcl(self, rcl):
        '''
        Select a city at random from rcl.
        Return city index in self.matrix
        
        Params:
        -------
        rcl: np.ndarray
            restricted candidate list
            vector of candidate city indexes.
        '''
        return self.rng.choice(rcl)
    
    
class GRASP:
    '''
    Greedy Randomised Adaptive Search Procedure algorithm
    for the Travelling Salesman Problem.
    
    
    The class has the following properties
    .best: float
        the best cost
        
    .best_solution: np.ndarray
        the best tour found
    
    '''
    def __init__(self, constructor, local_search, max_iter=1000,
                 time_limit=np.inf):
        '''
        Constructor
        
        Parameters:
        ---------
        constructor: object
            semi-greedy construction heuristic
            
        local_search: object
            local search heuristic e.g. `HillClimber`
            
        max_iter: int, optional (default=1000)
            The maximum number of iterations (restarts) of GRASP
            
        time_limit: float64, optional (default=np.inf)
            The maximum allowabl run time.
            
        
        '''
        # semi greedy tour construction method
        self.constructor = constructor
        
        # local search procedure
        self.local_search = local_search
        
        # max runtime budget for GRASP
        self.max_iter = max_iter
        self.time_limit = time_limit
        
        # init solution 
        self.best_solution = None
        self.best = None
    
        
    def solve(self):
        '''
        Run GRASP
        
        Returns:
        -------
        None
        '''
        
        self.best_solution = None
        self.best = -np.inf
        
        i = 0
        start = time.time()
    
        while i < self.max_iter and ((time.time() - start) < self.time_limit):
            
            i += 1
            
            # construction phase
            solution = self.constructor.build()
            
            # Improve solution via local search
            self.local_search.set_init_solution(solution)
            self.local_search.solve()
            
            current_solution = self.local_search.best_solutions[0]
            current = self.local_search.best_cost
            
            # check if better than current solution
            if current > self.best:
                self.best = current
                self.best_solution = current_solution
    

    

    