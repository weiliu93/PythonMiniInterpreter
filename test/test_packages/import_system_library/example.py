import os

print(os.path.join("a", "b"))

os_module = __import__("os")
print(os_module.path.join("c", "d"))
