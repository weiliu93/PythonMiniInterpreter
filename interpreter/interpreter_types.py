import dis


class Frame(object):
    """
    Simulate PyFrameObject
    """

    def __init__(self, f_code, *, f_globals=None, f_locals=None, f_back=None):
        self.f_code = f_code
        # instruction sequence
        self.instruction_sequence = InstructionSequence(f_code)
        # next instruction should be executed
        self.f_lasti = 0
        # data stack
        self.stack = []
        # data's level
        self.level_stack = []
        # previous frame
        self.f_back = f_back
        # namespaces
        self.f_globals = f_globals if f_globals else {}
        self.f_locals = f_locals if f_locals else {}
        if f_back:
            self.f_builtins = f_back.f_builtins
        else:
            if "__builtins__" in self.f_locals:
                self.f_builtins = self.f_locals["__builtins__"]
                if hasattr(self.f_builtins, "__dict__"):
                    self.f_builtins = self.f_builtins.__dict__
                assert isinstance(self.f_builtins, dict)
            else:
                self.f_builtins = {}
        # return value
        self.return_value = None
        # blocks
        self.blocks = []
        # TODO closure(free and cell)


    def stack_size(self):
        return len(self.stack)

    def top(self):
        return self.stack[-1]

    def top_level(self):
        return self.level_stack[-1]

    def topn(self, n):
        return self.stack[-n:]

    def pop(self):
        result = self.stack.pop()
        self.level_stack.pop()
        return result

    def popn(self, n):
        # cache last n elements
        result = self.stack[-n:]
        # remove last n elements from stacks
        self.stack = self.stack[0:-n]
        self.level_stack = self.level_stack[0:-n]
        return result

    def push(self, obj):
        self.stack.append(obj)
        self.level_stack.append(self.current_block_level())

    def is_completed(self):
        return self.f_lasti == len(self.instruction_sequence)

    def current_block_level(self):
        return len(self.blocks)


class Function(object):
    def __init__(
        self, func_name, func_code, func_global, func_defaults=None, func_closure=None
    ):
        self.func_name = func_name
        self.func_code = func_code
        self.func_global = func_global
        self.func_defaults = func_defaults
        self.func_closure = func_closure


class Block(object):
    def __init__(self, type, target, level):
        self.type = type
        self.target = target
        self.level = level


class Iterator(object):
    def __init__(self, frame, vm):
        self.frame = frame
        self.vm = vm

    def __iter__(self):
        return self

    def next(self):
        retval = self.vm.resume_frame(self.frame)
        # all instructions are executed
        if self.frame.is_completed():
            raise StopIteration("Iterator reach its last instruction")
        else:
            return retval

    __next__ = next


class Instruction(object):
    def __init__(self, op_name, op_arg):
        self.op_name = op_name
        self.op_arg = op_arg

    def __repr__(self):
        return "({}: {})".format(self.op_name, self.op_arg)

    def __str__(self):
        return self.__repr__()


class InstructionSequence(object):
    def __init__(self, code_object):
        self._instructions = []
        # parse instructions
        self._parse_instructions(code_object)

    def __repr__(self):
        return ", ".join(map(str, self._instructions))

    def __str__(self):
        return self.__repr__()

    def __iter__(self):
        for instruction in self._instructions:
            yield instruction

    def __getitem__(self, item):
        assert isinstance(item, int)
        if item < 0 or item >= len(self._instructions):
            raise IndexError(
                "Instruction Sequence Index `{}` Out Of Bound.".format(item)
            )
        return self._instructions[item]

    def __len__(self):
        return len(self._instructions)

    def _parse_instructions(self, code_object):
        code = code_object.co_code
        i, arg = 0, 0
        while i < len(code):
            op_name = dis.opname[code[i]]
            arg += code[i + 1]
            if op_name == "EXTENDED_ARG":
                arg <<= 8
                # EXTENDED_ARG is only a placeholder, keep instructions' relative order and offset unchanged
                self._instructions.append(Instruction(op_name, code[i + 1]))
            else:
                self._instructions.append(Instruction(op_name, arg))
                arg = 0
            i += 2
