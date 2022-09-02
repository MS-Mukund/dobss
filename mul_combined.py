import pulp as p
import numpy as np
import random
import sys

# Multiple LP problems
LOW = 20
UPP = 20  

J = 2
I = 2

if len(sys.argv) >= 4:
    I = int(sys.argv[2])
    J = int(sys.argv[3])
L = 3

# pr is a list of size l - probabilities of each follower
pr = [ random.uniform(0, 1) for _ in range(L) ]
pr = [ pr[l]/sum(pr) for l in range(L) ]
# pr = [ 0.47742417078005894, 0.5225758292199411 ]

# Payoff matrices
# Leader
R = np.array([[[ random.uniform(0, LOW) for _ in range(J) ] for _ in range(I) ] for _ in range(L) ]) 
# R = [ [ 3.85, 4.68 ], [ 3.46, 0.65 ] ], [ [ 4.18, 1.43 ], [ 3.48, 1.09 ] ]

# Follower
C = np.array([[[ random.uniform(0, UPP) for _ in range(J) ] for _ in range(I) ] for _ in range(L)])
# C = [ [ 4.48, 1.84 ], [ 4.42, 2.55 ] ], [ [ 4.6, 1.4 ], [ 3.24, 2.95 ] ]

# Probabilities
Probabs = [ ]
Rewards = [ ]
L_Rew   = [ ]

for j1 in range(J):
    for j2 in range(J):
        for j3 in range(J):
            mul_lp = p.LpProblem("LP_" + str(j1) + "_" + str(j2) + "_" + str(j3), p.LpMaximize)
            chanc = np.array([ p.LpVariable("Prob_" + str(i), 0, 1, p.LpContinuous ) for i in range(I) ])

            # objective 
            mul_lp += p.lpSum( chanc[i]*(pr[0]*R[0][i][j1] + pr[1]*R[1][i][j2] + pr[2]*R[2][i][j3] ) for i in range(I) ), "objective"

            # constraints
            mul_lp += p.lpSum( chanc[i] for i in range(I) ) == 1, "probability_sum"
            for k1 in range(J):
                for k2 in range(J):
                    for k3 in range(J):
                        mul_lp += p.lpSum( chanc[ct]*( pr[0]*C[0][ct][j1] + pr[1]*C[1][ct][j2] + pr[2]*C[2][ct][j3] ) for ct in range(I) ) >= p.lpSum( chanc[ct]*( pr[0]*C[0][ct][k1] + pr[1]*C[1][ct][k2] + pr[2]*C[2][ct][k3] ) for ct in range(I) ), "follower_" + str(k1) + "_" + str(k2) + "_" + str(k3)
            # for lt in range(L):
                # for k1 in range(J):
                    # mul_lp += p.lpSum( chanc[ct]*C[lt][ct][j1] for ct in range(I) ) >= p.lpSum( chanc[ct]*C[lt][ct][k1] for ct in range(I) )
            # for k1 in range(J):
            #     # mul_lp += p.lpSum( chanc[ct]*( pr[0]*C[0][ct][j1] + pr[1]*C[1][ct][j2] ) for ct in range(I) ) >= p.lpSum( chanc[ct]*( pr[0]*C[0][ct][k1] + pr[1]*C[1][ct][k2] ) for ct in range(I) ), "follower_" + str(k1) + "_" + str(k2)
            #     mul_lp += p.lpSum( chanc[ct]*( C[0][ct][j1] ) for ct in range(I) ) >= p.lpSum( chanc[ct]*( pr[0]*C[0][ct][k1] ) for ct in range(I) ), "follower_" + str(k1) 
            #     mul_lp += p.lpSum( chanc[ct]*( C[1][ct][j2] ) for ct in range(I) ) >= p.lpSum( chanc[ct]*( pr[1]*C[1][ct][k1] ) for ct in range(I) ), "follower2_" + str(k1)

            status = mul_lp.solve()
            if( p.LpStatus[status].lower() == 'optimal' ):
                Probabs.append( [ float(chanc[i].varValue) for i in range(I) ] )
                rew = sum( float(chanc[i].varValue)*( pr[0]*C[0][i][j1] + pr[1]*C[1][i][j2] + pr[2]*C[2][i][j3] ) for i in range(I) )
                Rewards.append(rew)
                L_Rew.append( p.value(mul_lp.objective) )
            else:
                Probabs.append( [ -1 for _ in range(I) ] )
                Rewards.append( -1 )
                L_Rew.append( -1 ) 

# for j1 in range(J):
#     mul_lp = p.LpProblem("LP_" + str(j1), p.LpMaximize)
#     chanc = np.array([ p.LpVariable("Prob_" + str(i), 0, 1, p.LpContinuous ) for i in range(I) ])

#     # objective 
#     mul_lp += p.lpSum( chanc[i]*(pr[0]*R[0][i][j1] ) for i in range(I) ), "objective"

#     # constraints
#     mul_lp += p.lpSum( chanc[i] for i in range(I) ) == 1, "probability_sum"
#     for k1 in range(J):
#         mul_lp += p.lpSum( chanc[ct]*( pr[0]*C[0][ct][j1] ) for ct in range(I) ) >= p.lpSum( chanc[ct]*( pr[0]*C[0][ct][k1] ) for ct in range(I) ), "follower_" + str(k1)

#     status = mul_lp.solve()
#     if( p.LpStatus[status].lower() == 'optimal' ):
#         Probabs.append( [ float(chanc[i].varValue) for i in range(I) ] )
#         rew = sum( float(chanc[i].varValue)*( pr[0]*C[0][i][j1] ) for i in range(I) )
#         Rewards.append(rew)
#         L_Rew.append( p.value(mul_lp.objective) )
#     else:
#         Probabs.append( [ -1 for _ in range(I) ] )
#         Rewards.append( -1 )
#         L_Rew.append( -1 )       

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

prob = p.LpProblem("DOBSS", p.LpMaximize)

M = 1e7
# 2 strategies for police 
# 2 strategies for each of the follower types
z = [ np.array([ [ p.LpVariable("z_" + str(l) + "_" + str(i) + "_" + str(j), 0, 1, p.LpContinuous ) for j in range(J) ] for i in range(I) ]) for l in range(L) ]
q = [ [ p.LpVariable("q_" + str(l) + "_" + str(j), 0, 1, p.LpInteger ) for j in range(J) ] for l in range(L) ]
a = [ p.LpVariable("a_" + str(l), lowBound=0 ) for l in range(L) ]

for l in range(L):
    prob += p.lpSum( z[l][i][j] for j in range(J) for i in range(I) ) == 1, "z_" + str(l) + "_sum"

# for l in range(L):
    # for i in range(I):
        # prob += p.lpSum( z[l][i][j] for j in range(J) ) <= 1, "z1_" + str(i) + "_" + str(l) + "_sum"
# for l in range(L):
    # for j in range(J):
        # prob += p.lpSum( z[l][i][j] for i in range(I) for l in range(L) ) <= 1, "z2_" + str(j) + "_" + str(l) + "_sum"
for l in range(L):
        for j in range(J):
            prob += q[l][j] <= p.lpSum( z[l][i][j] for i in range(I) ), "q_" + str(l) + "_" + str(j) + "_sum"
for i in range(I):
    for l in range(L):    
        prob += p.lpSum( z[l][i][j] for j in range(J) ) == p.lpSum( z[0][i][j] for j in range(J) ), "z3_" + str(i) + "_" + str(l) + "_sum"

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
    if ( p.LpStatus[status].lower() == 'optimal' ):
        f.write( str(max(L_Rew)) + '\n' )
        f.write( str(p.value(prob.objective)) + '\n')
    else:
        f.write( '-1\n' )
        f.write( '-1\n')

with open('b.txt', 'a+') as f:
    for l in range(L):
        for i in range(I):
            f.write(str(sum( float(z[l][i][j].varValue) for j in range(J))) + ' ' )
    f.write('\n') 

with open('vars.txt', 'a+') as f:
    for l in range(L):
        for i in range(I):
            for j in range(J):
                f.write(str( round(R[l][i][j], 2) ) + ',' + str( round(C[l][i][j], 2) ) + ' ' )
            f.write('\n')
        f.write('\n')
    
    f.write(str(pr[0]) + ' ' + str(pr[1]))   
    f.write('\n\n')