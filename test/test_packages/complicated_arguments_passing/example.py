def func(a , b , c = 10 , *args, **kwargs):
    print(a + b * c)
    for value in args:
        print(value)
    for key , value in kwargs.items():
        print(key, value)

func(1 , 3 , 100, 20, 30 , name = 'haha', pwd = 'hehe')