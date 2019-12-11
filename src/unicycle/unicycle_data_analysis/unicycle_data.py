import random
import numpy as np
import crocoddyl
crocoddyl.switchToNumpyArray()
try:
   from six.moves import cPickle
except:
   import pickle as cPickle



class unicycle_data():
    
    """
    Class to generate data from crocoddyl.
    
    """
    def __init__(self, n_trajectories: int = 10000, maxiter: int = 100, state_weight: float = 4., control_weight: float = 0.3, nodes: int = 30):
        """
        @ Description:
           n_trajectories = number of trajectories
           state, control weights = Cost weight
           nodes = number of knot points
           maxiter = maximum allowed itertations
        """
        self.n_trajectories = n_trajectories
        self.state_weight = state_weight
        self.control_weight = control_weight
        self.knots = nodes
        self.maxiter = maxiter
        
    def generate_trajectories_random_xyz(self, save: bool = False):
        


        starting_configurations = []
        optimal_trajectories = []
        controls = []

        for _ in range(self.n_trajectories):
            initial_config = [random.uniform(-2.1, 2.1), random.uniform(-2.1, 2.1), random.uniform(0, 1)]            
            model = crocoddyl.ActionModelUnicycle()

            model.costWeights = np.matrix([self.state_weight, self.control_weight]).T

            problem = crocoddyl.ShootingProblem(np.matrix(initial_config).T, [ model ] * self.knots, model)
            ddp = crocoddyl.SolverDDP(problem)
            ddp.solve([], [], self.maxiter)


        if ddp.isFeasible:
            state = np.array(ddp.xs)
            control = np.array(ddp.us)                
            starting_configurations.append(state[0,:])
            optimal_trajectories.append(state[1:,:])
            controls.append(control)


        if save:
            f = open('./data/variable_initial_state/starting_configurations.pkl', 'wb')
            cPickle.dump(starting_configurations, f, protocol=cPickle.HIGHEST_PROTOCOL)
            
            g = open("./data/variable_initial_state/optimal_trajectories.pkl", "wb")
            cPickle.dump(optimal_trajectories, g, protocol=cPickle.HIGHEST_PROTOCOL)

            h = open("./data/variable_initial_state/control.pkl", "wb")
            cPickle.dump(controls, h, protocol=cPickle.HIGHEST_PROTOCOL)
            
            f.close(), g.close(), h.close()


        else: 
            return starting_configurations, optimal_trajectories, controls


    def generate_trajectories_random_xy(self, theta_value:int = 0, save: bool = False):


        starting_configurations = []
        optimal_trajectories = []
        controls = []
        
        for _ in range(self.n_trajectories):
            initial_config = [random.uniform(-2.1, 2.1), random.uniform(-2.1, 2.1), theta_value]            
            model = crocoddyl.ActionModelUnicycle()
            
            model.costWeights = np.matrix([self.state_weight, self.control_weight]).T
            
            problem = crocoddyl.ShootingProblem(np.matrix(initial_config).T, [ model ] * self.knots, model)
            ddp = crocoddyl.SolverDDP(problem)
            ddp.solve([], [], self.maxiter)
            

            if ddp.isFeasible:
                state = np.array(ddp.xs)
                control = np.array(ddp.us)                
                starting_configurations.append(state[0,:])
                optimal_trajectories.append(state[1:,:])
                controls.append(control)


        if save:
            f = open('./data/constant_theta/starting_configurations.pkl', 'wb')
            cPickle.dump(starting_configurations, f, protocol=cPickle.HIGHEST_PROTOCOL)
            
            g = open("./data/constant_theta/optimal_trajectories.pkl", "wb")
            cPickle.dump(optimal_trajectories, g, protocol=cPickle.HIGHEST_PROTOCOL)

            h = open("./data/constant_theta/control.pkl", "wb")
            cPickle.dump(controls, h, protocol=cPickle.HIGHEST_PROTOCOL)
            
            f.close(), g.close(), h.close()




        else: 
            return starting_configurations, optimal_trajectories, controls
        
    def generate_trajectories_random_xz(self, y_value:int = -2, save: bool = False):


        starting_configurations = []
        optimal_trajectories = []
        controls = []
        
        for _ in range(self.n_trajectories):
            initial_config = [random.uniform(-2.1, 2.1), y_value, random.uniform(0,1)]            
            model = crocoddyl.ActionModelUnicycle()
            
            model.costWeights = np.matrix([self.state_weight, self.control_weight]).T
            
            problem = crocoddyl.ShootingProblem(np.matrix(initial_config).T, [ model ] * self.knots, model)
            ddp = crocoddyl.SolverDDP(problem)
            ddp.solve([], [], self.maxiter)
            

            if ddp.isFeasible:
                state = np.array(ddp.xs)
                control = np.array(ddp.us)                
                starting_configurations.append(state[0,:])
                optimal_trajectories.append(state[1:,:])
                controls.append(control)


        if save:
            f = open('./data/constant_y/starting_configurations.pkl', 'wb')
            cPickle.dump(starting_configurations, f, protocol=cPickle.HIGHEST_PROTOCOL)
            
            g = open("./data/constant_y/optimal_trajectories.pkl", "wb")
            cPickle.dump(optimal_trajectories, g, protocol=cPickle.HIGHEST_PROTOCOL)

            h = open("./data/constant_y/control.pkl", "wb")
            cPickle.dump(controls, h, protocol=cPickle.HIGHEST_PROTOCOL)
            
            f.close(), g.close(), h.close()



        else: 
            return starting_configurations, optimal_trajectories, controls
        
        
    def generate_trajectories_random_yz(self, x_value:int = -2, save: bool = False):
        

        starting_configurations = []
        optimal_trajectories = []
        controls = []
        
        for _ in range(self.n_trajectories):
            initial_config = [x_value, random.uniform(-2.1, 2.1), random.uniform(0,1)]            
            model = crocoddyl.ActionModelUnicycle()
            
            model.costWeights = np.matrix([self.state_weight, self.control_weight]).T
            
            problem = crocoddyl.ShootingProblem(np.matrix(initial_config).T, [ model ] * self.knots, model)
            ddp = crocoddyl.SolverDDP(problem)
            ddp.solve([], [], self.maxiter)
            

            if ddp.isFeasible:
                state = np.array(ddp.xs)
                control = np.array(ddp.us)                
                starting_configurations.append(state[0,:])
                optimal_trajectories.append(state[1:,:])
                controls.append(control)


        if save:
            f = open('./data/constant_x/starting_configurations.pkl', 'wb')
            cPickle.dump(starting_configurations, f, protocol=cPickle.HIGHEST_PROTOCOL)
            
            g = open("./data/constant_x/optimal_trajectories.pkl", "wb")
            cPickle.dump(optimal_trajectories, g, protocol=cPickle.HIGHEST_PROTOCOL)

            h = open("./data/constant_x/control.pkl", "wb")
            cPickle.dump(controls, h, protocol=cPickle.HIGHEST_PROTOCOL)
            
            f.close(), g.close(), h.close()



        else: 
            return starting_configurations, optimal_trajectories, controls
        
        
    def generate_similar_trajectories(self, x_value:int = -2, save: bool = False):
        """
        This is to check Carlos' point
        """

        pass
if __name__ == "__main__":    
    unicycle = unicycle_data()
    unicycle.generate_trajectories_random_xyz(save = True)
    unicycle.generate_trajectories_random_xy(save = True)
    unicycle.generate_trajectories_random_xz(save = True)
    unicycle.generate_trajectories_random_yz(save = True)

        
        
        
    
