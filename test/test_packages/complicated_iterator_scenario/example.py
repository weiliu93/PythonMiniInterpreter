def func(a, b):
    for _ in range(10):
        yield a + b
        a += 1
        b += 2


for value in func(1, b=4):
    print(value)
