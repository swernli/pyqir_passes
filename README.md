# PyQIR Passes

This repository has an example of using [PyQIR](https://github.com/qir-alliance/pyqir) to write simple analysis and transformation passes for QIR programs. Using some visitor-style patterns in base classes and overriding the desired functionality, you can write a new analysis pass or transformation pass for a quantum program with just Python. A few notes on the limitations of this approach:

- The `SimpleClonerPass` defined here makes use of the PyQIR convenience functions for generating a new module with the contents of the original, and as such does not handle arbitrary LLVM IR but rather focuses on the subset that is recognized by PyQIR (as of v0.10.5).
- Module flags are not generally preserved by the cloner, such that the transformed program uses flags and attributed consistent with how a module would look when generated from scratch by PyQIR.
- Changing properties such as the number of qubits or results used by a program requires extra steps when initializing the `SimpleModule` used for cloning, which means a pass that changes these values will need to override `on_module` and avoid calling `super().on_module` (see `ExampleQubitShiftPass` for this pattern).

The pass_tests.pyi file has a short test that shows how two passes can be used together to analyze and transform an input QIR program.

For more informtion on PyQIR, check out the repository at [https://github.com/qir-alliance/pyqir](https://github.com/qir-alliance/pyqir).

For more inforation on the Quantum Intermediate Representation (QIR), check out the QIR Specification at [https://github.com/qir-alliance/qir-spec](https://github.com/qir-alliance/qir-spec).
