import matplotlib.pyplot as plt
import subprocess

open('a.txt', 'w').close()
open('b.txt', 'w').close()

for i in range(20):
    subprocess.run(["python", "combined.py" ])

with open('a.txt', 'r') as f:
    lines = f.readlines()
    lines = [float(line.strip()) for line in lines]

    mult_lp = [ lines[ 2*i ] for i in range(len(lines)//2) ]
    dobss   = [ lines[ 2*i + 1 ] for i in range(len(lines)//2) ]

    plt.xticks(range(1, 21))
    
    plt.plot( [ i+1 for i in range(20)], mult_lp, 'r', label='Multiple LP')
    plt.plot( [ i+1 for i in range(20)], dobss,  'b', label='DOBSS')
    plt.legend()

    # plt.show()
    plt.savefig('plot.png')