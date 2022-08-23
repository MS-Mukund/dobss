import pulp as p
import numpy as np
import random
import sys

# Multiple LP problems

LOW = 20
UPP = 20    

J = 3
I = 4

if len(sys.argv) >= 4:
    I = int(sys.argv[2])
    J = int(sys.argv[3])

L = 1

# Payoff matrices
# Leader
R = np.array([[[ random.uniform(0, LOW) for _ in range(J) ] for _ in range(I) ] for l in range(L) ]) 
# R = np.array([ [ 0.04, 0.16 ], [ 4.79, 1.9 ] ])

# Follower
C = np.array([[[ random.uniform(0, UPP) for _ in range(J) ] for _ in range(I) ] for l in range(L)])
# C = np.array([ [ 4.81, 0.25 ], [ 3.92, 4.57 ] ])

# Probabilities
Probabs = [ ]
Rewards = [ ]
L_Rew   = [ ]

l = 0
for j in range(J):
    mul_lp = p.LpProblem("LP_" + str(j), p.LpMaximize)
    chanc = np.array([ p.LpVariable("Prob_" + str(j) + "_" + str(i), 0, 1, p.LpContinuous ) for i in range(I) ])

    # objective 
    mul_lp += p.lpSum( chanc[i]*R[l][i][j] for i in range(I) ), "objective_" + str(j)

    # constraints
    mul_lp += p.lpSum( chanc[i] for i in range(I) ) == 1, "Prob_" + str(j) + "_sum"
    for k in range(J):
        mul_lp += p.lpSum( chanc[ct]*( C[l][ct][j] ) for ct in range(I) ) >= p.lpSum( chanc[ct]*( C[l][ct][k] ) for ct in range(I) ), "follower_" + str(j) + "_" + str(k)

    status = mul_lp.solve()
    if( p.LpStatus[status].lower() == 'optimal' ):
        Probabs.append( [ float(chanc[i].varValue) for i in range(I) ] )
        rew = sum( Probabs[j][i]*C[l][i][j] for i in range(I) )
        Rewards.append(rew)
        L_Rew.append( p.value(mul_lp.objective) )
    else:
        Probabs.append( [ -1 for _ in range(I) ] )
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
        for j in range(J):
            f.write('-1 ')
    else: 
        for j in range(len(Probabs[ind])):
            f.write(str(Probabs[ind][j]) + ' ')
    
    f.write('\n')

# with open('b.txt', 'a+') as f:
    # for i in range(J):
        # for j in range(len(Probabs[i])):
            # f.write(str(Probabs[i][j]) + ' ')
        # f.write('\n')
# 
    # f.write(str(L_Rew) + '\n\n')

prob = p.LpProblem("DOBSS", p.LpMaximize)

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
        prob += 0 <= a[l] - p.lpSum( C[l][i][j]*(p.lpSum(z[l][i][h] for h in range(J))) for i in range(I) ), "left_a_" + str(l) + "_" + str(j)
        prob += a[l] - p.lpSum( C[l][i][j]*(p.lpSum(z[l][i][h] for h in range(J))) for i in range(I) ) <= (1 - q[l][j])*M, "right_a_" + str(l) + "_" + str(j)

# objective
prob += p.lpSum( pr[l]*R[l][i][j]*z[l][i][j] for l in range(L) for i in range(I) for j in range(J) ), "objective"

status = prob.solve()
# print(f"solution status: {p.LpStatus[status]}")
# print(f"objective value: {p.value(prob.objective)}")  

with open('a.txt', 'a+') as f:
    # f.write( 'Multiple LP: ' + str(max(L_Rew)) + '\n' )
    # f.write( 'DOBSS: ' + str(p.value(prob.objective)) + '\n')
    f.write( str(max(L_Rew)) + '\n' )
    f.write( str(p.value(prob.objective)) + '\n')

with open('b.txt', 'a+') as f:
    for i in range(I):
        f.write(str(sum(z[l][i][j].varValue for l in range(L) for j in range(J))) + ' ' )
    f.write('\n') 

with open('vars.txt', 'a+') as f:
    for i in range(I):
        for j in range(J):
            f.write(str( round(R[l][i][j], 2) ) + ',' + str( round(C[l][i][j], 2) ) + ' ' )
        f.write('\n')
    
    f.write('\n')
