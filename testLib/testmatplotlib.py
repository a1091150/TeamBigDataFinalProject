# -*- coding: utf-8 -*-
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import numpy as np
import random
import math

mydata = [random.randrange(1,101,1) for _ in range (10)]
print mydata
plt.hist(mydata)
plt.show()
