import pulp as p
import numpy as np
import random

# Multiple LP problems

num_f_types = 1
l_rew_ubound = 5
f_rew_ubound = 5    

num_f_act = 2
num_l_act = 2

# Payoff matrices
# Leader
R = np.array([ [ random.uniform(0, l_rew_ubound) for _ in range(num_f_act) ] for _ in range(num_l_act) ]) 
# R = np.array([ [ 2, 4 ], [ 1, 3 ] ])

# Follower
C = np.array([ [ random.uniform(0, f_rew_ubound) for _ in range(num_f_act) ] for _ in range(num_l_act) ])
# C = np.array([ [ 1, 0 ], [ 0, 1 ] ])

# Probabilities
Probabs = [ ]
Rewards = [ ]
L_Rew   = [ ]

for j in range(num_f_act):
    mul_lp = p.LpProblem("LP_" + str(j), p.LpMaximize)
    chanc = np.array([ p.LpVariable("Prob_" + str(j) + "_" + str(i), 0, 1, p.LpContinuous ) for i in range(num_l_act) ])

    # constraints
    mul_lp += p.lpSum( chanc[i] for i in range(num_l_act) ) == 1, "Prob_" + str(j) + "_sum"
    for k in range(num_l_act):
        mul_lp += p.lpSum( chanc[ct]*( C[j][ct] - C[k][ct] ) for ct in range(num_l_act) ) >= 0, "follower_" + str(j) + "_" + str(k)

    mul_lp += p.lpSum( chanc[i]*R[i][j] for i in range(num_l_act) ), "objective_" + str(j)

    status = mul_lp.solve()
    if( p.LpStatus[status].lower() == 'optimal' ):
        Probabs.append( [ float(chanc[i].varValue) for i in range(num_l_act) ] )
        rew = sum( Probabs[j][i]*C[i][j] for i in range(num_l_act) )
        Rewards.append(rew)
        L_Rew.append( p.value(mul_lp.objective) )
    else:
        Probabs.append( [ -1 for _ in range(num_l_act) ] )
        Rewards.append( -1 )
        L_Rew.append( -1 )    

# print(f"Leader's reward = {max(L_Rew)}")
ind = -1
for i, rew in enumerate(L_Rew):
    if( rew == max(L_Rew) ):
        ind = i
        break
   
with open('b.txt', 'a+') as f:
    if ( max(L_Rew) < 0 ):
        for j in range(num_f_act):
            f.write( '-1 ' )
    else: 
        for j in range(len(Probabs[ind])):
            f.write(str(Probabs[ind][j]) + ' ')
    
    f.write('\n')

# with open('b.txt', 'a+') as f:
    # for i in range(num_f_act):
        # for j in range(len(Probabs[i])):
            # f.write(str(Probabs[i][j]) + ' ')
        # f.write('\n')

prob = p.LpProblem("DOBSS", p.LpMaximize)

I, J, L = 2, 2, 1
R_upper, F_upper = 20, 20
A_upper = F_upper*I + 1

M = 1e7
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
        prob += 0 <= a[l] - p.lpSum( C[i][j]*(p.lpSum(z[l][i][h] for h in range(J))) for i in range(I) ), "left_a_" + str(l) + "_" + str(j)
        prob += a[l] - p.lpSum( C[i][j]*(p.lpSum(z[l][i][h] for h in range(J))) for i in range(I) ) <= (1 - q[l][j])*M, "right_a_" + str(l) + "_" + str(j)

# objective
prob += p.lpSum( pr[l]*R[i][j]*z[l][i][j] for l in range(L) for i in range(I) for j in range(J) ), "objective"

status = prob.solve()
# print(f"solution status: {p.LpStatus[status]}")
# print(f"objective value: {p.value(prob.objective)}")  

with open('a.txt', 'a+') as f:
    # f.write( 'Multiple LP: ' + str(max(L_Rew)) + '\n' )
    # f.write( 'DOBSS: ' + str(p.value(prob.objective)) + '\n')
    f.write( str(max(L_Rew)) + '\n' )
    f.write( str(p.value(prob.objective)) + '\n')

with open('b.txt', 'a+' ) as f:
    for i in range(I):
        f.write(str(sum(z[l][i][j].varValue for l in range(L) for j in range(J))) + ' ' )
    f.write('\n')      

# for v in prob.variables():
    # print(f"{v.name} = {v.varValue}")
