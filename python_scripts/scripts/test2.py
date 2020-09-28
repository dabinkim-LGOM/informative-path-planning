

class Sample():
    def __init__(self):
        self.a = 1.0
        self.b = 2.0 

    def ret(self):
        return self.a 

sa = Sample()
a = sa.ret()
a = 5 
print(sa.a)