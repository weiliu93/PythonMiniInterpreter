class A(object):
    def func(self, value):
        return value * 3


class B(A):
    def b_func(self, value):
        print(value - self.func(value))


b = B()
b.b_func(7)
