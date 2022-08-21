import pulp as p
import numpy as np
import random

prob = p.LpProblem("DOBSS", p.LpMaximize)

I, J, L = 3, 3, 1
R_upper, F_upper = 20, 20
A_upper = F_upper*I + 1

M = 1e7
# R is a l-membered list of 2D arrays of size (no. of possible leader strategies, no. of possible follower strategies) where l is the number of follower types
R = [ np.array([ [ random.uniform(0, R_upper) for _ in range(J) ] for _ in range(I) ]) for _ in range(L) ]

# C - payoff for follower strategies
C = [ np.array([ [ random.uniform(0, F_upper) for _ in range(J) ] for _ in range(I) ]) for _ in range(L) ]

# 2 strategies for police 
# 2 strategies for each of the follower types
z = [ np.array([ [ p.LpVariable("z_" + str(l) + "_" + str(i) + "_" + str(j), 0, 1, p.LpContinuous ) for j in range(J) ] for i in range(I) ]) for l in range(L) ]
q = [ [ p.LpVariable("q_" + str(l) + "_" + str(j), 0, 1, p.LpInteger ) for j in range(J) ] for l in range(L) ]
a = [ p.LpVariable("a_" + str(l), lowBound=0 ) for l in range(L) ]

# pr is a list of size l - probabilities of each follower
pr = [ random.uniform(0, 1) for _ in range(L) ]
pr = [ pr[l]/sum(pr) for l in range(L) ]

for l in range(L):
    prob += p.lpSum( z[l][i][j] for j in range(J) for i in range(I) ) == 1, "z_" + str(l) + "_sum"
for l in range(L):
        for j in range(J):
            prob += q[l][j] <= p.lpSum( z[l][i][j] for i in range(I) ), "q_" + str(l) + "_" + str(j) + "_sum"
for l in range(L):
    prob += p.lpSum( q[l][j] for j in range(J) ) == 1, "q_sum_" + str(l)
for l in range(L):
    for j in range(J):
        prob += 0 <= a[l] - p.lpSum( C[l][i][j]*(p.lpSum(z[l][i][h] for h in range(J))) for i in range(I) ), "left_a_" + str(l) + "_" + str(j)
        prob += a[l] - p.lpSum( C[l][i][j]*(p.lpSum(z[l][i][h] for h in range(J))) for i in range(I) ) <= (1 - q[l][j])*M, "right_a_" + str(l) + "_" + str(j)

# testing
# for l in range(L):
#     for i in range(I):
#         for j in range(J):
#             prob +=  z[l][i][j] >= 0, "z_" + str(l) + "_" + str(i) + "_" + str(j) + "_2"

# objective
prob += p.lpSum( pr[l]*R[l][i][j]*z[l][i][j] for l in range(L) for i in range(I) for j in range(J) ), "objective"

print(prob)
status = prob.solve()
print(f"solution status: {p.LpStatus[status]}")
print(f"objective value: {p.value(prob.objective)}")   

for v in prob.variables():
    print(f"{v.name} = {v.varValue}")
