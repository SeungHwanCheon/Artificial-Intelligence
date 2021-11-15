import pandas as pd

lottery = pd.read_csv('lottery.csv')

for i in range (1,46) :

    cnt = 0

    for j in ['first', 'second','third','fourth','fifth','sixth','bonus'] :

        for k in range (959) :

            if i == lottery[j][k] :

                cnt = cnt + 1

    print (i, "->", cnt, "times")


