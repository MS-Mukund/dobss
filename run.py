import matplotlib.pyplot as plt
import subprocess

for ctl in range(2, 5):
    for ctf in range(3, 5):
        open('a.txt', 'w').close()
        open('b.txt', 'w').close()
        open('vars.txt', 'w').close()

        for _ in range(20):
            subprocess.run(["python", "mul_combined.py", str(ctl), str(ctf) ])

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
            plt.savefig('plots/plot_1_' + str(ctl) + '_' + str(ctf) + '.png')
            plt.clf()