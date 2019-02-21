class VirtualMachineParsingException(Exception):
    def __init__(self, message="Virtual Machine Parsing Exception"):
        super().__init__(message)


class VirtualMachineInvalidInstructionException(Exception):
    def __init__(self, message="Virtual Machine receive invalid instruction"):
        super().__init__(message)
