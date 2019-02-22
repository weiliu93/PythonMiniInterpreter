def func():
    value = 1
    for _ in range(10):

        def inner_func():
            print(value)

        yield inner_func
        value += 2


for f in func():
    f()
