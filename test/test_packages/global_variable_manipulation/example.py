value = 10


def func():
    global value
    value = 100


print(value)

func()

print(value)
