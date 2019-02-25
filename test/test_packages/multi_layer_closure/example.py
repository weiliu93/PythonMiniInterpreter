def func():
    value_list = []

    def f():
        print(value_list)

        def inner_f(value):
            value_list.append(value)

        return inner_f

    f()(10)
    f()(20)
    f()(0)


func()
