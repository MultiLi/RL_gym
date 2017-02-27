import numpy as np

class Exp_Replay():
    def __init__(self, num, s_dim ,a_dim):
        self.pointer = -1
        self.size = num
        self.length = 0
        self.s = np.zeros([num] + s_dim)
        self.a = np.zeros(num,dtype = int)
        self.r = np.zeros(num)
        self.s1 = np.zeros([num] + s_dim)

    def insert(self,s,a,r,ns):
        self.pointer += 1
        if self.pointer == self.size:
            self.pointer = 0
        self.s[self.pointer] = s
        self.a[self.pointer] = a
        self.r[self.pointer] = r
        self.s1[self.pointer] = ns
        if self.length < self.size:
            self.length += 1