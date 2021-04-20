'''
Greedy Randomised Adaptive Search Procedure

classes and functions.
'''

import numpy as np
import time

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
    

    
class MonitoredLocalSearch:
    '''
    Extends a local search class and provides the observer pattern.
    An external object can observe the local search object and catch the
    termination event (end of local search).  The observer is notified and
    passed the results of the local search.

    Use cases: 
    ----------
    In GRASP this is useful for an algorithm sizing the RCL and learning 
    on average how different sizes of RCL perform.
    '''
    def __init__(self, local_search):
        '''
        Constructor:
        
        Params:
        ------
        local_search: Object
            Must implement .solve(), best_cost, best_solution
        '''
        self.local_search = local_search
        self.observers = []
    
    def register_observer(self, observer):
        '''
        register an object to observe the local search
        
        The observer should implement 
        local_search_terminated(*args, **kwargs)
        '''
        self.observers.append(observer)
        
    def set_init_solution(self, solution):
        '''
        Set the initial solution
        
        Params:
        --------
        solution: np.ndarray
            vector representing the initial solution
        '''
        self.local_search.set_init_solution(solution)
    
    def solve(self):
        '''
        Run the local search.  
        At the end of the run all observers are notified.
        '''    
        # run local search
        self.local_search.solve()
        
        # notify observers after search terminates.
        best = self.local_search.best_cost
        solution = self.local_search.best_solutions[0]
        self.notify_observers(best, solution)
        
    
    def notify_observers(self, *args, **kwargs):
        '''
        Observers must implement `local_search_terminated()`
        method.
        
        Params:
        ------
        *args: list
            variable number of arguments
            
        **kwargs: dict
            key word arguments
        '''
        for o in self.observers:
            o.local_search_terminated(*args, **kwargs)
    
    def _get_best_cost(self):
        '''
        best cost from internal local_search object
        '''
        return self.local_search.best_cost
    
    def _get_best_solutions(self):
        '''
        get best solutions from local_search object
        '''
        return self.local_search.best_solutions
    
    best_cost = property(_get_best_cost, doc='best cost')
    best_solutions = property(_get_best_solutions, doc='best solution')
    
    
class ReactiveRCLSizer:
    '''
    Dynamically update the probability of selecting a 
    value of r for the size of the RCL.
    
    Implements Reactive GRASP.
    
    '''
    def __init__(self, r_list, local_search, freq=None, random_seed=None):
        '''
        Constructor
        
        Params:
        -------
        r_list: list
            vector of sizes for RCL e.g. [1, 2, 3, 4, 5]
            
        local_search: MonitoredLocalSearch
            local_search to monitor
            
        freq: int, optional (default=None)
            Frequency in iterations at which the probabilities are updated.
            When set to None it defaults to the length of r_list * 2
            
        random_seed: int, optional (default=None)
            Control random sampling for reproducible result
        '''
        # list of r sizes
        self.r_list = r_list
        
        # set of indexes to work with probabilities
        self.elements = np.arange(len(r_list))
    
        # probability of choosing r (initially uniform)
        self.probs = np.full(len(r_list), 1/len(r_list))
        
        # mean performance of size r
        self.means = np.full(len(r_list), 1.0)
        
        # runs of size r
        self.allocations = np.full(len(r_list), 0)
        
        # local search to monitor
        self.local_search = local_search
        
        # frequency of updating probs
        if freq is None:
            self.freq = len(self.r_list)
        else:
            self.freq = freq
        
        # number of iterations within frequency
        self.iter = 0
        
        # current r index
        self.index = -1
        
        # to init run one of each r value
        self.init = True
        
        # imcumbent solution cost
        self.best_cost = -np.inf
        
        # register sizer as observer of the local search
        local_search.register_observer(self)
        
        # random no. gen
        self.rng = np.random.default_rng(random_seed)
    
    def local_search_terminated(self, *args, **kwargs):
        '''
        Termination of the local search
        '''
        # iteration complete
        self.iter += 1
        
        # get the best cost found in the iteration
        iter_cost = args[0]

        # record iteration took plaxe with index i
        self.allocations[self.index] += 1
        
        # update running mean
        mean_x = self.means[self.index]
        n = self.allocations[self.index]
        self.means[self.index] += (iter_cost - mean_x) / n
        
        self.update_r()
        
        # update incumbent cost if required
        if iter_cost > self.best_cost:
            self.best_cost = iter_cost
        
        # update probs if freq met.
        if self.iter >= self.freq and not self.init:
            self.iter = 0
            self.update_probability()
            
        
    def update_probability(self):
        '''
        Let $q_i = f^* / A_i$
        and $p_i = `\dfrac{q_i}{\sum_{j=1}^{m} q_j}$
        
        where
        
        $f^*$ is the incumbent (cost)
        $A_i$ is the mean cost found with r_i
        
        larger q_i indicates more suitable values of r_i
        '''
        q = self.best_cost / self.means
        self.probs = q / q.sum()
    
    def update_r(self):
        '''
        update the size of r
        
        Note that the implementation ensures that all r values are run
        for at least one iteration of the algorithm.
        '''
        # initial bit of logic makes sure there is at least one run of all probabilities
        if self.init:
            self.index += 1
            if self.index >= len(self.r_list):
                self.init = False
                self.index = self.rng.choice(self.elements, p=self.probs)
        else:
            self.index = self.rng.choice(self.elements, p=self.probs)
    
    def get_size(self):
        '''
        Return the selected size of the RCL
        
        The selection is done using a discrete distribution
        self.r_probs.
        '''
        return self.r_list[self.index]
    
    
class RandomPlusGreedyConstructor(SemiGreedyConstructor):
    '''
    Random + semi-greedy construction of a tour.
    
    The first n cities of a tour are randomly constructed.  
    The remaining cities are seleted using the standard semi-greedy approach.
    
    For a city i creates a restricted candidate list of size r
    i.e the r shortest distances from city i.  Next city is chosen
    with equal probability. 
    
    Repeats until tour is constructed.
    
    '''
    def __init__(self, rcl_sizer, tour, matrix, p_rand=0.2,
                 random_seed=None):
        '''
        RandomPlusGreedy Constructor method
        
        Params:
        ------
        rcl_sizer: object
            sizes the restricted candidate list
        
        tour: np.ndarray
            vector of city indexes included in problem
            
        matrix: np.ndarray
            matrix of travel costs
            
        p_rand: float, optional (default=0.2)
            Proportion of tour that is randomly constructed
            
        random_seed: int
            used to control sampling provides a
            reproducible result.
        '''
        
        # super class init
        super().__init__(rcl_sizer, tour, matrix,
                       random_seed)
        
        # proportion of tour that is randomly constructed
        self.p_rand = p_rand
        self.n_rand = int(p_rand * len(tour))
        self.n_greedy = len(tour) - self.n_rand - 1
        
    
    def build(self):
        '''
        Random followed by semi-greedy contruction of tour
        
        Returns:
        --------
        np.array
        '''
        # first city in tour
        solution = np.array([self.tour[0]])    
        # next n_rand cities are random
        rand = self.rng.choice(self.tour[1:], size=self.n_rand, replace=False)
        solution = np.append(solution, rand)
        
        # remaining cities are semi-greedy
        for i in range(self.n_greedy):
            r = self.rcl_sizer.get_size()
            rcl = self.get_rcl(r, solution, solution[-1])
            next_city = self.random_from_rcl(rcl)
            solution = np.append(solution, np.array([next_city]))
            
        return solution        

    
class ConstructorWithMemory:
    '''
    Provides a construction heuristic with a short term memory
    '''
    def __init__(self, constructor, memory_size=100):
        '''Constructor method
        
        Params:
        -------
        constructor: Object
            Implements build() and returns a solution
            
        memory_size, int, optional (default=100)
            size of tabu list
        '''
        self.constructor = constructor
        self.memory_size = memory_size
        # memory implemented as list
        self.history = []
        
    def build(self):
        '''
        Run the stochastic construction heuristic
        
        Re-runs heuristic if results is within memory
        
        Returns:
        --------
        np.ndarray
        '''
        solution = self.constructor.build()
        while str(solution) in self.history:
            solution = self.constructor.build()
        
        # if at capacity remove oldest solution
        if len(self.history) >= self.memory_size: 
            self.history.pop(0)
        
        self.history.append(str(solution))
        return solution
    
class EliteSet:
    '''
    Tracks and updates an elite set of solutions produced by a local search.
    '''
    def __init__(self, local_search=None, max_size=10, min_delta=0):
        '''
        Constructor
        Params:
        -------
        local_search: MonitoredLocalSearch
            The local search that produces candidates for the elite
            set.
        
        max_size: int, optional (default=10)
            maximum entries in the elite set

        min_delta: int, optional (Default=0)
            The min cardinality difference between tours to allow entry
            E.g. a = [1, 2, 3, 4, 5]; b = [1, 3, 4, 2, 5]. delta = 3.
            Vary delta > 0 to increase diversity (but may limit entry)
        '''
        if local_search is not None:
            self.local_search = local_search
            local_search.register_observer(self)
        
        self.min_delta = min_delta
        self.max_size = max_size
        
        # data structures for elite solutions
        self.solutions = None
        self.costs = None
        
        self.n_updates = 0
    
    @property
    def is_empty(self):
        return self.solutions is None
    
    def is_elite(self, solution):
        '''
        Is the solution a member of the elite set
        
        Params:
        ------
        solution: np.ndarray
            TSP solutution
            
        Returns:
        --------
        bool
        '''
        if self.solutions is None:
            return False
        else:
            result = np.where((self.solutions==solution).all(axis=1))[0]
            return len(result) > 0
    
    def local_search_terminated(self, *args, **kwargs):
        ''''
        Termination of the local search
        '''
        s = args[1]
        s_cost = args[0]
        self.update(s, s_cost)
            
    def init_elite_set(self, s, s_cost):
        '''
        Initalise the elite set
        '''
        self.solutions = np.array([s])
        self.costs = np.array([s_cost])
    
    def update(self, s, s_cost):
        '''
        Update the elite set to maximise performance and diversity

        Params:
        -------
        s: np.ndarray
            TSP tour

        s_cost: float
            TSP tour cost

        Returns:
        -------
        Tuple: np.ndarray, np.ndarray
            elite_set, elite_costs
        '''
        if self.solutions is None:
            self.init_elite_set(s, s_cost)
        
        elif len(self.solutions) < self.max_size:
            delta = (s != self.solutions).sum(axis=1).min()
            if delta > self.min_delta:
                self.solutions = np.append(self.solutions, [s], axis=0)
                self.costs = np.append(self.costs, [s_cost], axis=0)
        else:
            # aim to increase performance and diversity of elite set
            worst_cost = self.costs.min()
            delta = (s != self.solutions).sum(axis=1).min()

            if s_cost > worst_cost and delta > self.min_delta:
                replace_idx = self.costs.argmin()
                self.solutions[replace_idx] = s
                self.costs[replace_idx] = s_cost
                self.n_updates += 1
                
    def get_best_solution(self):
        '''Returns the best solution and cost
        from the elite set'''
        best_idx = np.array(self.costs).argmax()
        return self.solutions[best_idx], self.costs[best_idx]
    
class TSPPathRelinker:
    '''
    Path relinking for basic symmetric TSP.
    
    Performs forward or backward relinking with a random
    elite solution from the elite set.
    
    After PR is complete a local search is used to take the solution
    to a local optimum.
    '''
    def __init__(self, local_search, tracker, objective, 
                 trunc=None, random_seed=None):
        '''
        Constructor
        
        local_search: object
            local search algorithm
            
        tracker: EliteSet
            The elite set tracker object
            
        objective: object
            TSP Objective
            
        trunc: int, optional (default=None)
            Truncate the relinking to a fixed number of steps
            
        random_seed: int, optional (default=None)
            used to control sampling from elite set
        '''
        
        self.local_search = local_search
        self.tracker = tracker
        self.obj = objective
        self.trunc = trunc
        
        # used for selecting from the elite set at random
        self.rng = np.random.default_rng(random_seed)
        
        self.calls = 0
        self.made_it = 0
        self.already_elite = 0
    
    def evaluate_neighbour(self, from_city, current, guiding):
        '''
        Performs a two city swap and returns the cost.

        Params:
        ------
        from_city: int
            The city to swap

        current: np.ndarray
            The current solution

        guiding: np.ndarray
            The guiding solution

        obj: Object
            The TSP objective function

        Returns:
        -------
        float
        '''
        from_idx = np.where(current==from_city)[0][0]
        to_idx = np.where(guiding==from_city)[0][0]
        # swap and evaluate
        current[from_idx], current[to_idx] = current[to_idx], current[from_idx]
        cost= self.obj.evaluate(current)

        #swap back
        current[from_idx], current[to_idx] = current[to_idx], current[from_idx]
        return cost
    
    def next_restricted_neighbour_move(self, current, guiding):
        '''
        Iteratively search through a restricted neighbourhood based 
        on the guiding solution and greedily selects the best move.

        Params:
        -------
        current: np.ndarray
            The current solution

        guiding: np.ndarray
            The guiding solution

        obj: Object
            The TSP objective function

        Returns:
        -------
        (np.ndarray, float)
        (best_solution, best_cost) 
        '''
        # cities in the restricted neighbourhood

        swaps = current[current != guiding]
        
        costs = np.full(len(swaps), -np.inf)
        i = 0
        # is there a way to eliminate the python loop?
        for from_city in swaps:
            # evaluate all swaps in current restricted neighbourhood
            costs[i] = self.evaluate_neighbour(from_city, current, guiding)
            i += 1

        best_index = np.argmax(costs)
        from_city = swaps[best_index]

        # get index of cities in current and guiding solutions.
        from_idx = np.where(current==from_city)[0][0]
        to_idx = np.where(guiding==from_city)[0][0]
        # swap and evaluate
        current[from_idx], current[to_idx] = current[to_idx], current[from_idx]
        return current, costs[best_index]
    
    def relink(self, solution, elite_solution=None, direction='backward'):
        '''
        Relink between a current and guiding solution
        
        Params:
        -------
        solution: np.ndarray
            a none elite set TSP solution
            
        elite_solution: np.ndarray.
            guiding TSP solution
            
        direction: str
             forward: move from current towards guiding
             backward: swap current and guiding and relink.
        '''
        
        # catch local optimum solutions that are already elite
        if self.tracker.is_elite(solution):
            self.already_elite += 1
            return solution, self.obj.evaluate(solution)
        
        # random elite solution if none specified
        if elite_solution is None:
            elite_solution = self.rng.choice(self.tracker.solutions)
        
        # if 'forward' selected.
        if direction == 'forward':
            current, guiding = solution, elite_solution
        elif direction == 'backward':
            current, guiding = elite_solution, solution
        else:
            raise ValueError('Please select backward of forward relinking')
        
        # moves to relink n / 2
        n_moves = len(current[current != guiding]) // 2

        # used to truncate path relinking
        if self.trunc is not None:
            if self.trunc > n_moves:
                raise ValueError(f'@trunc must be <= moves to relink {n_moves}')
            else:
                n_moves = self.trunc

        # path relinking
        costs = [self.obj.evaluate(solution)]
        solutions = [solution]
        for i in range(n_moves):
            current, cost = self.next_restricted_neighbour_move(current.copy(), 
                                                                guiding)
            solutions.append(current)
            costs.append(cost)

        # best solution and cost
        # error here if the relinking has found no improvements.
        best_idx = np.array(costs).argmax()
        pr_solution, pr_cost =  solutions[best_idx], costs[best_idx]
        
        # local search on relinked solution...
        self.local_search.set_init_solution(pr_solution)
        self.local_search.solve()
        
        # return local optimum
        return self.local_search.best_solutions[0], self.local_search.best_cost
    
    
class GRASPPlusPathRelinking:
    '''
    Greedy Randomised Adaptive Search Procedure algorithm
    hybridised with Path Relinking for the Travelling Salesman Problem
    '''
    def __init__(self, constructor, local_search, relinker,
                 tracker, max_iter=1000, time_limit=np.inf):
        
        # semi greedy tour construction method
        self.constructor = constructor
        
        # local search procedure
        self.local_search = local_search
        
        # path relinker
        self.path_relinker = relinker
        
        # elite solution tracker
        self.elite_tracker = tracker
        
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
            current_cost = self.local_search.best_cost
            
            if not self.elite_tracker.is_empty:
                # backwards path relinking with elite solution selected at random
                current_solution, current_cost = self.path_relinker.relink(current_solution)
            
            # added because in theory best solution might not go into Elite set due to diversity
            if current_cost > self.best:
                self.best = current_cost
                self.best_solution = current_solution
            
            # update elite solutions
            self.elite_tracker.update(current_solution, current_cost)