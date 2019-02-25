class A(object):
    def __init__(self):
        self.name = "haha"
        self.id = "hehe"

    def print(self):
        print("name is {}, id is {}".format(self.name, self.id))


a = A()
a.print()
