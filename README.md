The deeper_networks branch has crocoddyl devel version
The master branch has robotpkg-py36-crocoddyl version


Optimal_learning: Pinocchio and Crocoddyl.

Initial research for Crocoddyl.

1: Rohans' thesis

2: How important is it to go to global optimum

3: Optimizing with a log barrier. Given a function, f(x) add a log(f(x)) in optimization.

4: Optmization is mostly inverting matrices. And most matrices are just 90% full of zeroes. So adding, subtracting, multiplying zeroes leads to waste of time. Sparsity-> matrices full of zeroes. Markovian and non Markovian problems and transforming a Markovian problem into non Markovian problems. Sparsity of cost matrices

5: https://www.stat.cmu.edu/~ryantibs/convexopt-S15/scribes/

6: Augmented lagrangian methods.

7: IDMM algorithms

8: Crocoddyl is a multiple shooting algorithm. Read multiple ShootingProblem
