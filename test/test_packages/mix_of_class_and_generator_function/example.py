class A(object):
    def func(self):
        for i in range(10):
            if i % 2 == 1:
                yield i


a = A()
for value in a.func():
    print(value)
