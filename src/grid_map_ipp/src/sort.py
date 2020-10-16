import numpy as np
import math 

center = [50, 50]
x = [51, 43, 42, 50, 59, 61, 54, 44]
y = [60, 55, 46, 40, 43, 53, 60, 58]
thet = []
for i in range(len(x)):
    value = math.atan2(y[i]-center[1], x[i]-center[0])
    thet.append(value)
print(thet)

for i in range(len(x)):
    for j in range(len(x)-i-1):
        if thet[j] < thet[j+1]:
            tmp = thet[j]
            thet[j] = thet[j+1]
            thet[j+1] = tmp

print(thet)