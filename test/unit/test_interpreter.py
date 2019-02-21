import sys
import os
import io

sys.path.append(
    os.path.join(
        os.path.join(os.path.dirname(__file__), os.pardir, os.pardir), "interpreter"
    )
)

from mini_interpreter import VirtualMachine

test_packages = os.path.join(os.path.dirname(__file__), os.pardir, "test_packages")


def test_print_hello_world():
    string_output = io.StringIO()
    vm = VirtualMachine()
    vm.set_std_stream(vm_out=string_output)
    vm.run_compiled_code('print("hello")')
    vm.reset_std_stream()
    assert string_output.getvalue() == "hello\n"


def test_complicated_iterator_scenario():
    package_path = os.path.join(test_packages, "complicated_iterator_scenario")
    test_file = os.path.join(package_path, "example.py")

    string_output = io.StringIO()
    vm = VirtualMachine()
    vm.set_std_stream(vm_out=string_output)
    vm.run_compiled_code(open(test_file).read())
    vm.reset_std_stream()

    assert string_output.getvalue().split() == list(map(str, [5, 8, 11, 14, 17, 20, 23, 26, 29, 32]))
