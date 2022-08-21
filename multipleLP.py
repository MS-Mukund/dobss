''' 
Maximise: sigma(p_s*u_l) for all s in S
Subject to:
    sigma(p_s) = 1
    sigma(p_s*u_f(s,t)) >= sigma(p_s*u_f(s,t')) for t in T
'''

import pulp as p
import numpy as np
import random

# Multiple LP problems

num_f_types = 1
l_rew_ubound = 20
f_rew_ubound = 20

num_f_act = 3
num_l_act = 3

# Payoff matrices
# Leader
R = np.array([ [ random.uniform(0, l_rew_ubound) for _ in range(num_f_act) ] for _ in range(num_l_act) ]) 

# Follower
C = np.array([ [ random.uniform(0, f_rew_ubound) for _ in range(num_f_act) ] for _ in range(num_l_act) ])

# Probabilities
Probabs = [ ]
Rewards = [ ]
L_Rew   = [ ]

for j in range(num_f_act):
    prob = p.LpProblem("LP" + str(j), p.LpMaximize)
    var = [ p.LpVariable("Prob_" + str(j) + "_" + str(i), 0, 1, p.LpContinuous ) for i in range(num_l_act) ]

    prob += p.lpSum( var[i] for i in range(num_l_act) ) == 1, "Prob_" + str(j) + "_sum"
    prob += p.lpSum( var[i]*R[i][j] for i in range(num_l_act) ), "Prob_" + str(j) + "_obj"

    prob.solve()
    Probabs.append( [ var[i].varValue for i in range(num_l_act) ] )
    
    rew = sum( Probabs[j][i]*C[i][j] for i in range(num_l_act) )
    Rewards.append(rew)
    L_Rew.append( p.value(prob.objective) )

    print(f"Probability {j} = {Probabs[j]}")
    print(f"Reward {j} = {Rewards[j]}")
    print()

ind = -1
cur_max = -1
for j in range(len(Rewards)):
    if Rewards[j] > cur_max:
        ind = j
        cur_max = Rewards[j]

print(f"Leader's reward = {L_Rew[ind]}")
