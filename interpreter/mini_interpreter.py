"""
basic functionality, support common operators and load, store
"""
from interpreter_types import Frame, Function, Block, Iterator
from interpreter_exceptions import (
    VirtualMachineParsingException,
    VirtualMachineInvalidInstructionException,
)

from functools import partial

import ast
import sys
import operator
import logging

logger = logging.getLogger(__name__)


class VirtualMachine(object):
    """
    Python 3.6 simple interpreter
    """

    def __init__(self):
        self._frames = []
        self.backup_stdin = sys.stdin
        self.backup_stdout = sys.stdout

    def run_compiled_code(self, code_obj):
        code_obj = self._parse_python_object(code_obj)
        f_locals = f_globals = {
            "__builtins__": __builtins__,
            "__name__": code_obj.co_name,
            "__doc__": None,
            "__package__": None,
        }
        frame = self.make_frame(code_obj, f_globals, locals=f_locals)
        retval = self.run_frame(frame)
        assert frame.stack_size() == 0
        return retval

    def make_frame(
        self,
        code_obj,
        globals,
        locals=None,
        positional_arguments=None,
        kw_arguments=None,
        defaults=None,
        f_back=None,
    ):
        locals = locals if locals else {}
        positional_arguments = positional_arguments if positional_arguments else []
        kw_arguments = kw_arguments if kw_arguments else {}
        defaults = defaults if defaults else []
        total_args = code_obj.co_argcount
        default_offset = total_args - len(defaults)

        # Use defaults to initialize local namespace first
        for index in range(default_offset, total_args):
            arg_name = code_obj.co_varnames[index]
            locals[arg_name] = defaults[index - default_offset]
        # Then use position arguments
        for index in range(len(positional_arguments)):
            arg_name = code_obj.co_varnames[index]
            locals[arg_name] = positional_arguments[index]
        # Finally use keyword arguments
        for keyword_name, arg in kw_arguments.items():
            if keyword_name not in locals:
                locals[keyword_name] = arg
            else:
                raise VirtualMachineInvalidInstructionException(
                    "Argument Re-assignment in positional argument and keyword argument"
                )

        frame = Frame(code_obj, f_locals=locals, f_globals=globals, f_back=f_back)
        return frame

    def resume_frame(self, frame):
        frame.back = self._fetch_current_frame()
        retval = self.run_frame(frame)
        frame.back = None
        return retval

    def run_frame(self, frame):
        assert isinstance(frame, Frame)
        total_number_of_frame = len(self._frames)

        # push frame to stack
        self._frames.append(frame)
        # dispatch instructions for current frame
        self.dispatch(frame)
        # pop frame from stack
        self._frames.pop()

        # total number of frames should not be changed
        assert len(self._frames) == total_number_of_frame
        return frame.return_value

    def dispatch(self, frame):
        why = None
        while frame.f_lasti < len(frame.instruction_sequence):
            next_instruction = frame.instruction_sequence[frame.f_lasti]
            # Compare operations
            if next_instruction.op_name.startswith("COMPARE"):
                why = self.exec_COMPARE_OPERATIONS(frame, next_instruction)
            elif next_instruction.op_name.startswith("UNARY"):
                why = self.exec_UNARY_OPERATIONS(frame, next_instruction)
            else:
                why = getattr(
                    self,
                    "exec_{}".format(next_instruction.op_name),
                    self.exec_ILLEGAL_INSTRUCTION,
                )(frame, next_instruction)
            if why:
                # break, exception, etc.
                break
        return why

    def set_std_stream(self, vm_in=None, vm_out=None):
        if vm_in:
            sys.stdin = vm_in
        if vm_out:
            sys.stdout = vm_out

    def reset_std_stream(self):
        sys.stdin = self.backup_stdin
        sys.stdout = self.backup_stdout

    """
    --------------------------------------Exception------------------------------------
    """

    def exec_ILLEGAL_INSTRUCTION(self, frame, instruction):
        raise VirtualMachineInvalidInstructionException(
            "Illegal instruction received, `{}`".format(instruction.op_name)
        )

    """
    ---------------------------------Name manipulation---------------------------------
    """

    def exec_LOAD_CONST(self, frame, instruction):
        const_object = frame.f_code.co_consts[instruction.op_arg]
        frame.push(const_object)
        frame.f_lasti += 1

    def exec_LOAD_NAME(self, frame, instruction):
        name = frame.f_code.co_names[instruction.op_arg]
        if name in frame.f_locals:
            frame.push(frame.f_locals[name])
        elif name in frame.f_globals:
            frame.push(frame.f_globals[name])
        elif name in frame.f_builtins:
            frame.push(frame.f_builtins[name])
        else:
            raise VirtualMachineInvalidInstructionException(
                "Load name `{}` failed".format(name)
            )
        frame.f_lasti += 1

    def exec_STORE_NAME(self, frame, instruction):
        name = frame.f_code.co_names[instruction.op_arg]
        value = frame.pop()
        frame.f_locals[name] = value
        frame.f_lasti += 1

    def exec_LOAD_FAST(self, frame, instruction):
        name = frame.f_code.co_varnames[instruction.op_arg]
        frame.push(frame.f_locals[name])
        frame.f_lasti += 1

    def exec_STORE_FAST(self, frame, instruction):
        name = frame.f_code.co_varnames[instruction.op_arg]
        frame.f_locals[name] = frame.pop()
        frame.f_lasti += 1

    def exec_RETURN_VALUE(self, frame, instruction):
        frame.return_value = frame.pop()
        frame.f_lasti += 1

    def exec_LOAD_ATTR(self, frame, instruction):
        obj = frame.pop()
        name = frame.f_code.co_names[instruction.op_arg]
        frame.push(getattr(obj, name))
        frame.f_lasti += 1

    def exec_STORE_ATTR(self, frame, instruction):
        target, obj = frame.popn(2)
        name = frame.f_code.co_names[instruction.op_arg]
        setattr(obj, name, target)
        frame.f_lasti += 1

    def exec_LOAD_GLOBAL(self, frame, instruction):
        name = frame.f_code.co_names[instruction.op_arg]
        # check global namespace first
        if name in frame.f_globals:
            obj = frame.f_globals[name]
            frame.push(obj)
            frame.f_lasti += 1
        # then check builtin namespace
        elif name in frame.f_builtins:
            obj = frame.f_builtins[name]
            frame.push(obj)
            frame.f_lasti += 1
        else:
            raise VirtualMachineInvalidInstructionException(
                "Load global failed, attribute name is not found in global and builtin namespace"
            )

    """
    ------------------------------Control flow statements------------------------------
    """

    def exec_POP_JUMP_IF_FALSE(self, frame, instruction):
        if not frame.pop():
            # Every instruction contains two byte, so we can divide arg by two all the time
            frame.f_lasti = instruction.op_arg // 2
        else:
            frame.f_lasti += 1

    def exec_POP_JUMP_IF_TRUE(self, frame, instruction):
        if frame.pop():
            frame.f_lasti = instruction.op_arg // 2
        else:
            frame.f_lasti += 1

    def exec_JUMP_FORWARD(self, frame, instruction):
        frame.f_lasti += instruction.op_arg // 2
        frame.f_lasti += 1

    def exec_JUMP_ABSOLUTE(self, frame, instruction):
        frame.f_lasti = instruction.op_arg // 2

    """
    ---------------------------------Import statements---------------------------------
    """

    def exec_IMPORT_NAME(self, frame, instruction):
        level, from_list = frame.popn(2)
        name = frame.f_code.co_names[instruction.op_arg]
        current_frame = self._fetch_current_frame()
        frame.push(
            __import__(
                name, current_frame.f_globals, current_frame.f_locals, from_list, level
            )
        )
        frame.f_lasti += 1

    def exec_IMPORT_FROM(self, frame, instruction):
        mod = frame.pop()
        name = frame.f_code.co_names[instruction.op_arg]
        frame.push(getattr(mod, name))
        frame.f_lasti += 1

    def exec_IMPORT_STAR(self, frame, instruction):
        mod = frame.pop()
        if hasattr(mod, "__all__"):
            __all__ = set(getattr(mod, "__all__"))
            for attr in filter(lambda attr: attr in __all__, dir(mod)):
                frame.f_locals[attr] = getattr(mod, attr)
        else:
            for attr in dir(mod):
                frame.f_locals[attr] = getattr(mod, attr)
        frame.f_lasti += 1

    """
    --------------------------------Compare operations---------------------------------
    """

    def exec_COMPARE_OPERATIONS(self, frame, instruction):
        ops = [
            operator.lt,
            operator.le,
            operator.eq,
            operator.ne,
            operator.gt,
            operator.ge,
            partial(self.swap_operator, operator.contains),
            partial(self.inv_operator, partial(self.swap_operator, operator.contains)),
            operator.is_,
            operator.is_not,
        ]
        a, b = frame.popn(2)
        frame.push(ops[instruction.op_arg](a, b))
        frame.f_lasti += 1

    def inv_operator(self, op, a, b):
        """
        neg final result
        """
        return not op(a, b)

    def swap_operator(self, op, a, b):
        """
        swap arguments for op function
        """
        return op(b, a)

    """
    --------------------------------Unary operations----------------------------------
    """

    def exec_UNARY_OPERATIONS(self, frame, instruction):
        if instruction.op_name == "UNARY_NOT":
            value = frame.pop()
            frame.push(not value)
            frame.f_lasti += 1
        else:
            pass

    """
    --------------------------------Loop operations----------------------------------
    """

    def exec_SETUP_LOOP(self, frame, instruction):
        block = Block(
            "loop",
            frame.f_lasti + instruction.op_arg // 2,
            frame.current_block_level() + 1,
        )
        frame.blocks.append(block)
        frame.f_lasti += 1

    def exec_BREAK_LOOP(self, frame, instruction):
        current_block = frame.blocks[-1]
        frame.f_lasti = current_block.target

    def exec_YIELD_VALUE(self, frame, instruction):
        value = frame.top()
        frame.return_value = value
        frame.f_lasti += 1
        return "yield"

    def exec_GET_ITER(self, frame, instruction):
        obj = frame.pop()
        frame.push(iter(obj))
        frame.f_lasti += 1

    def exec_FOR_ITER(self, frame, instruction):
        iter_obj = frame.top()
        try:
            value = next(iter_obj)
            frame.push(value)
        except StopIteration:
            frame.pop()
            frame.f_lasti += instruction.op_arg // 2
        frame.f_lasti += 1

    def exec_POP_BLOCK(self, frame, instruction):
        # pop block first
        last_block = frame.blocks.pop()
        while frame.stack_size() > 0 and (frame.top_level() == last_block.level):
            frame.pop()
        frame.f_lasti += 1

    """
    --------------------------------Function operations--------------------------------
    """

    def exec_CALL_FUNCTION(self, frame, instruction):
        args = frame.popn(instruction.op_arg + 1)
        func, arguments = args[0], args[1:]
        if callable(func):
            # If it is not a special builtin function
            if not self.special_builtin_function_handler(
                frame, instruction, func, positional_arguments=arguments
            ):
                frame.push(func(*arguments))
                frame.f_lasti += 1
        else:
            CO_GENERATOR = 32
            new_frame = self.make_frame(
                func.func_code,
                frame.f_globals,
                positional_arguments=arguments,
                defaults=func.func_defaults,
                f_back=self._fetch_current_frame(),
            )
            # generator function
            if func.func_code.co_flags & CO_GENERATOR:
                frame.push(Iterator(new_frame, self))
            # normal function call
            else:
                self.run_frame(new_frame)
                frame.push(new_frame.return_value)
            frame.f_lasti += 1

    def exec_CALL_FUNCTION_KW(self, frame, instruction):
        keyword_names = frame.pop()
        args = frame.popn(instruction.op_arg)
        func = frame.pop()
        positional_arguments, keywords_arguments = [], {}
        keywords_arguments_start_index = instruction.op_arg - len(keyword_names)
        for i in range(instruction.op_arg):
            if i >= keywords_arguments_start_index:
                keywords_arguments[
                    keyword_names[i - keywords_arguments_start_index]
                ] = args[i]
            else:
                positional_arguments.append(args[i])
        if callable(func):
            # If it is not a special builtin function
            if not self.special_builtin_function_handler(
                frame,
                instruction,
                func,
                positional_arguments=positional_arguments,
                keyword_arguments=keywords_arguments,
            ):
                frame.push(func(*positional_arguments, **keywords_arguments))
                frame.f_lasti += 1
        else:
            CO_GENERATOR = 32
            new_frame = self.make_frame(
                func.func_code,
                frame.f_globals,
                positional_arguments=positional_arguments,
                kw_arguments=keywords_arguments,
                defaults=func.func_defaults,
                f_back=self._fetch_current_frame(),
            )
            # generator function
            if func.func_code.co_flags & CO_GENERATOR:
                frame.push(Iterator(new_frame, self))
            # normal function call
            else:
                self.run_frame(new_frame)
                frame.push(new_frame.return_value)
            frame.f_lasti += 1

    def special_builtin_function_handler(
        self,
        frame,
        instruction,
        func,
        positional_arguments=None,
        keyword_arguments=None,
    ):
        # special check for locals and globals, since they will refer to current env, not virtual env
        if func == locals:
            frame.push(frame.f_locals)
            frame.f_lasti += 1
            return True
        elif func == globals:
            frame.push(frame.f_globals)
            frame.f_lasti += 1
            return True
        else:
            return False

    def exec_MAKE_FUNCTION(self, frame, instruction):
        func_code_object, func_name = frame.popn(2)
        # TODO actually I don't know how MAKE_FUNCTION implemented..... Does 0, 1, 9 have any special meaning?
        if instruction.op_arg == 0:
            # Function without default argument values
            func = Function(func_name, func_code_object, frame.f_globals)
            frame.push(func)
        elif instruction.op_arg == 1:
            # Function with default arguments
            default_tuple = frame.pop()
            func = Function(
                func_name,
                func_code_object,
                frame.f_globals,
                func_defaults=default_tuple,
            )
            frame.push(func)
        elif instruction.op_arg == 9:
            # TODO closure manipulation

            pass
        else:
            raise VirtualMachineInvalidInstructionException(
                "Unknown make function arg `{}`".format(instruction.op_arg)
            )
        frame.f_lasti += 1

    """
    --------------------------------Binary Operations--------------------------------
    """

    def exec_BINARY_ADD(self, frame, instruction):
        value1, value2 = frame.popn(2)
        frame.push(value1 + value2)
        frame.f_lasti += 1

    def exec_BINARY_SUBTRACT(self, frame, instruction):
        value1, value2 = frame.popn(2)
        frame.push(value1 - value2)
        frame.f_lasti += 1

    def exec_BINARY_MULTIPLY(self, frame, instruction):
        value1, value2 = frame.popn(2)
        frame.push(value1 * value2)
        frame.f_lasti += 1

    def exec_BINARY_TRUE_DIVIDE(self, frame, instruction):
        value1, value2 = frame.popn(2)
        frame.push(value1 / value2)
        frame.f_lasti += 1

    def exec_BINARY_FLOOR_DIVIDE(self, frame, instruction):
        value1, value2 = frame.popn(2)
        frame.push(value1 // value2)
        frame.f_lasti += 1

    def exec_BINARY_SUBSCR(self, frame, instruction):
        obj, item = frame.popn(2)
        frame.push(obj[item])
        frame.f_lasti += 1

    def exec_BINARY_LSHIFT(self, frame, instruction):
        value, bit = frame.popn(2)
        frame.push(value << bit)
        frame.f_lasti += 1

    def exec_BINARY_RSHIFT(self, frame, instruction):
        value, bit = frame.popn(2)
        frame.push(value >> bit)
        frame.f_lasti += 1

    def exec_BINARY_MODULO(self, frame, instruction):
        value1, value2 = frame.popn(2)
        frame.push(value1 % value2)
        frame.f_lasti += 1

    """
    ---------------------------------Inplace operations---------------------------------
    """

    def exec_INPLACE_ADD(self, frame, instruction):
        value1, value2 = frame.popn(2)
        value1 += value2
        frame.push(value1)
        frame.f_lasti += 1

    def exec_INPLACE_SUBTRACT(self, frame, instruction):
        value1, value2 = frame.popn(2)
        value1 -= value2
        frame.push(value1)
        frame.f_lasti += 1

    def exec_INPLACE_MULTIPLY(self, frame, instruction):
        value1, value2 = frame.popn(2)
        value1 *= value2
        frame.push(value1)
        frame.f_lasti += 1

    def exec_INPLACE_TRUE_DIVIDE(self, frame, instruction):
        value1, value2 = frame.popn(2)
        value1 /= value2
        frame.push(value1)
        frame.f_lasti += 1

    def exec_INPLACE_FLOOR_DIVIDE(self, frame, instruction):
        value1, value2 = frame.popn(2)
        value1 //= value2
        frame.push(value1)
        frame.f_lasti += 1

    def exec_INPLACE_LSHIFT(self, frame, instruction):
        value1, value2 = frame.popn(2)
        value1 <<= value2
        frame.push(value1)
        frame.f_lasti += 1

    def exec_INPLACE_RSHIFT(self, frame, instruction):
        value1, value2 = frame.popn(2)
        value1 >>= value2
        frame.push(value1)
        frame.f_lasti += 1

    def exec_INPLACE_MODULO(self, frame, instruction):
        value1, value2 = frame.popn(2)
        value1 %= value2
        frame.push(value1)
        frame.f_lasti += 1

    """
    ----------------------------------Stack operations----------------------------------
    """

    def exec_POP_TOP(self, frame, instruction):
        frame.pop()
        frame.f_lasti += 1

    """
    --------------------------------Build data structure--------------------------------
    """

    def exec_BUILD_TUPLE(self, frame, instruction):
        args = frame.popn(instruction.op_arg)
        frame.push(tuple(args))
        frame.f_lasti += 1

    def exec_BUILD_LIST(self, frame, instruction):
        args = frame.popn(instruction.op_arg)
        frame.push(list(args))
        frame.f_lasti += 1

    def exec_BUILD_SET(self, frame, instruction):
        args = frame.popn(instruction.op_arg)
        frame.push(set(args))
        frame.f_lasti += 1

    def exec_BUILD_MAP(self, frame, instruction):
        args = frame.popn(instruction.op_arg << 1)
        keys, values = args[0::2], args[1::2]
        new_dict = {}
        for k, v in zip(keys, values):
            new_dict[k] = v
        frame.push(new_dict)
        frame.f_lasti += 1

    def exec_BUILD_CONST_KEY_MAP(self, frame, instruction):
        const_keys = frame.pop()
        values = frame.popn(instruction.op_arg)
        new_dict = {}
        for k, v in zip(list(const_keys), values):
            new_dict[k] = v
        frame.push(new_dict)
        frame.f_lasti += 1

    def exec_BUILD_SLICE(self, frame, instruction):
        args = frame.popn(instruction.op_arg)
        frame.push(slice(*args))
        frame.f_lasti += 1

    """
    --------------------------------Other helper functions--------------------------------
    """

    def _parse_python_object(self, obj):
        # source code is given
        if isinstance(obj, str):
            ast_root = ast.parse(obj)
            return compile(ast_root, "exec_code", "exec")
        # ast tree is given
        elif isinstance(obj, ast.Module):
            return compile(obj, "exec_code", "exec")
        else:
            # code object is given
            sample_code = "a = 10"
            if type(obj) == type(compile(ast.parse(sample_code), "sample", "exec")):
                return obj
            raise VirtualMachineParsingException(
                "Virtual machine parse given object failed"
            )

    def _fetch_current_frame(self):
        return None if not self._frames else self._frames[-1]


if __name__ == "__main__":
    code = open("./code.py").read()
    ast_root = ast.parse(code)
    code_object = compile(ast_root, "code", "exec")

    vm = VirtualMachine()
    vm.run_compiled_code(code_object)
