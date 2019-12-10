Each folder contains three separate files: starting_configurations , trajectory, control. 
Note that in Crocoddyl, ddp.xs contains T + 1 terms, while ddp.us contains T terms. The extra term in ddp.xs is the initial
starting point.

Here the "trajectory" stores ddp.xs sans the starting point.
The starting points are stored separately in "starting_configurations" while "control" stores the control forces.


              

Constant_theta :
                
                theta = 0
                
                x, y are randomly sampled from (-2.1, 2.1)
                
Constant_x :
                
                x = -2
                
                theta, y are randomly sampled from (0,1) and (-2.1, 2.1)
                
Constant_y :
                
                y = -2
                
                theta, x are randomly sampled from (0,1) and (-2.1, 2.1)
                
                
variable_initial_state :
                
                             
                x, y, theta, x are randomly sampled from (-2.1, 2.1), (-2.1, 2.1) and (0,1)
