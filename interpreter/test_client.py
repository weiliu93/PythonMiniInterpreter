import ast
import dis


ast_root = ast.parse(open('./test_code.py').read())

code_object = compile(ast_root, 'code', 'exec')

# dis.dis(code_object.co_consts[0])
dis.dis(code_object.co_consts[1])

# print(code_object.co_consts[0].co_flags)
# print(code_object.co_consts[0].co_argcount)

# import test_code
#
# f = test_code.func()
# print(f.__closure__[0].cell_contents)

# print(code_object.co_consts[0].co_consts[2].co_flags)