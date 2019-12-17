import random, numpy as np, crocoddyl
crocoddyl.switchToNumpyArray()
def iter_analysis():
    iters = []
    iters_500 = []
    iters_1000 = []
    for _ in range(10000):
        initial_config = [random.uniform(-2.1, 2.1), random.uniform(-2.1, 2.1), 0]            
        model = crocoddyl.ActionModelUnicycle()
        model.costWeights = np.matrix([4, 0.3]).T

        problem = crocoddyl.ShootingProblem(np.matrix(initial_config).T, [ model ] * 20, model)
        ddp = crocoddyl.SolverDDP(problem)
        ddp.solve([], [], 100 )
        iters.append(ddp.iter)
        
        ddp2 = crocoddyl.SolverDDP(problem)
        ddp2.solve([], [], 500 )
        iters_500.append(ddp2.iter)    
        ddp3 = crocoddyl.SolverDDP(problem)
        ddp3.solve([], [], 1000 )
        iters_1000.append(ddp3.iter)  

    del ddp, model, ddp2, ddp3    
    allowed_iterations_500 = []
    for _ in range(10000):
        initial_config = [random.uniform(-2.1, 2.1), random.uniform(-2.1, 2.1), 0]            
        model = crocoddyl.ActionModelUnicycle()
        model.costWeights = np.matrix([4, 0.3]).T

        problem = crocoddyl.ShootingProblem(np.matrix(initial_config).T, [ model ] * 20, model)
        ddp = crocoddyl.SolverDDP(problem)
        ddp.solve([], [], 500 )
        allowed_iterations_500.append(ddp.iter)    
    del model, ddp   
    allowed_iterations_1000 = []
    for _ in range(10000):
        initial_config = [random.uniform(-2.1, 2.1), random.uniform(-2.1, 2.1), 0]            
        model = crocoddyl.ActionModelUnicycle()
        model.costWeights = np.matrix([4, 0.3]).T

        problem = crocoddyl.ShootingProblem(np.matrix(initial_config).T, [ model ] * 20, model)
        ddp = crocoddyl.SolverDDP(problem)
        ddp.solve([], [], 1000 )
        allowed_iterations_1000.append(ddp.iter)  
    del ddp, model 
    failure = 0
    for i in zip(iters, iters_1000, iters_500):        
        if not (i[0] == i[1] == i[2]): failure += 1
    print("First:\n")        
    print(f"Variance of ddp.iter. For same initial points, ddp.iter remains largely the same \
    i.e. for a particular x, y, theta , ddp.solve([], [], 100/500/1000) gives the same ddp.iter. \nThis is expected, however, \
    does not give the same result in a few cases.\n \
    In this example it crocoddyl failed in {failure} instances over 10 k points.\n")
    
    print(".......................................................................\n")
    print("Second:\n")
    print("Randomly generate 10K points thrice. ")
    
    print(f"Allowed iterations : 100  |  Minimum: {min(iters)}       |  Maximum: {max(iters)}       | Average: {np.mean(iters)} |")
    print(f"Allowed iterations : 500  |  Minimum: {min(iters_500)}   |  Maximum: {max(iters_500)}   | Average: {np.mean(iters_500)} |")
    print(f"Allowed iterations : 1000 |  Minimum: {min(iters_1000)}  |  Maximum: {max(iters_1000)} | Average: {np.mean(iters_1000)} |")
    del iters, iters_500, iters_1000, allowed_iterations_500, allowed_iterations_1000
    
    
#iter_analysis()

