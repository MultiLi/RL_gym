import numpy as np

class Memory():
    def __init__(self, num):
        self.pointer = -1
        self.size = num
        self.length = 0
        self.s = np.zeros((num,4))
        self.a = np.zeros(num,dtype = int)
        self.r = np.zeros(num)
        self.s1 = np.zeros((num,4))

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
