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

    assert string_output.getvalue().split() == list(
        map(str, [5, 8, 11, 14, 17, 20, 23, 26, 29, 32])
    )


def test_complicated_arguments_passing():
    package_path = os.path.join(test_packages, "complicated_arguments_passing")
    test_file = os.path.join(package_path, "example.py")

    string_output = io.StringIO()
    vm = VirtualMachine()
    vm.set_std_stream(vm_out=string_output)
    vm.run_compiled_code(open(test_file).read())
    vm.reset_std_stream()

    target = ["301", "20", "30", "name", "haha", "pwd", "hehe"]
    assert string_output.getvalue().split() == target


def test_even_odd_loop():
    package_path = os.path.join(test_packages, "even_odd_loop")
    test_file = os.path.join(package_path, "example.py")

    string_output = io.StringIO()
    vm = VirtualMachine()
    vm.set_std_stream(vm_out=string_output)
    vm.run_compiled_code(open(test_file).read())
    vm.reset_std_stream()

    target = ["0", "2", "4", "6"]
    assert string_output.getvalue().split() == target


def test_closure_usage():
    package_path = os.path.join(test_packages, "loop_closure_usage")
    test_file = os.path.join(package_path, "example.py")

    string_output = io.StringIO()
    vm = VirtualMachine()
    vm.set_std_stream(vm_out=string_output)
    vm.run_compiled_code(open(test_file).read())
    vm.reset_std_stream()

    target = list(map(str, list(range(1, 20, 2))))
    assert string_output.getvalue().split() == target


def test_classic_closure_example():
    package_path = os.path.join(test_packages, "classic_closure_example")
    test_file = os.path.join(package_path, "example.py")

    string_output = io.StringIO()
    vm = VirtualMachine()
    vm.set_std_stream(vm_out=string_output)
    vm.run_compiled_code(open(test_file).read())
    vm.reset_std_stream()

    target = list(map(str, [21] * 10))
    assert string_output.getvalue().split() == target


def test_recursive_function_call():
    package_path = os.path.join(test_packages, "recursive_function_call")
    test_file = os.path.join(package_path, "example.py")

    string_output = io.StringIO()
    vm = VirtualMachine()
    vm.set_std_stream(vm_out=string_output)
    vm.run_compiled_code(open(test_file).read())
    vm.reset_std_stream()

    assert string_output.getvalue() == "89\n573147844013817084101\n"
