'''
NOTE: Doesn't work, since Pulp designed for linear programming only, not quadratic
'''

import pulp as p
# from qpsolvers import solve_qp as p
import numpy as np
import random

prob = p.LpProblem("DOBSS", p.LpMaximize)

I, J, L = 2, 2, 2
R_upper, F_upper = 20, 20
A_upper = F_upper*I + 1

M = 1e7
# R is a l-membered list of 2D arrays of size (no. of possible leader strategies, no. of possible follower strategies) where l is the number of follower types
R = [ np.array([ [ random.uniform(0, R_upper) for _ in range(J) ] for _ in range(I) ]) for _ in range(L) ]

# C - payoff for follower strategies
C = [ np.array([ [ random.uniform(0, F_upper) for _ in range(J) ] for _ in range(I) ]) for _ in range(L) ]

# 2 strategies for police 
# 2 strategies for each of the follower types
x = [ p.LpVariable("x_" + str(i), 0, 1, p.LpContinuous ) for i in range(I) ]
q = [ [ p.LpVariable("q_" + str(l) + "_" + str(j), 0, 1, p.LpInteger ) for j in range(J) ] for l in range(L) ]
a = [ p.LpVariable("a_" + str(l), lowBound=0 ) for l in range(L) ]

# pr is a list of size l - probabilities of each follower
pr = [ random.uniform(0, 1) for _ in range(L) ]
pr = [ pr[l]/sum(pr) for l in range(L) ]

prob += p.lpSum( x ) == 1, "x_sum"
prob += p.lpSum( q ) == 1, "q_sum"
for l in range(L):
    for j in range(J):
        prob += 0 <= a[l] - p.lpSum( C[l][i][j] * x[i] for i in range(I) ), "left_a_" + str(l) + "_" + str(j)
        prob += a[l] - p.lpSum( C[l][i][j] * x[i] for i in range(I) ) <= ( 1 - q[l][j] )*M , "right_a_" + str(l) + "_" + str(j)

# objective
prob += p.lpSum( pr[l]*R[l][i][j]*x[l]*q[l][j] for l in range(L) for i in range(I) for j in range(J) ), "objective"

print(prob)
status = prob.solve()
print(f"solution status: {p.LpStatus[status]}")
print(f"objective value: {p.value(prob.objective)}")   

for v in prob.variables():
    print(f"{v.name} = {v.varValue}")
