import numpy as np
import math as m


def isConvergent():
    conv = 0
    x = 2
    for i in range(1, 1000):
        conv = conv + 1/np.square(i)

    print(conv)



isConvergent()
