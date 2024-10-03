# Licensed under the MIT License.

# %%
import pyqir
from pyqir_passes import (
    ExampleResetPruningPass,
    ExampleInstructionCounterPass,
    as_qis_gate,
)

# Read the extra_resets.ll file from test_qir
with open("test_qir/extra_resets.ll", "r") as f:
    ll_code = f.read()
mod = pyqir.Module.from_ir(pyqir.Context(), ll_code)

# Count up the number of reset instructions before transformation
before = ExampleInstructionCounterPass(
    lambda instr: "reset" in as_qis_gate(instr).get("gate", "")
)
before.on_module(mod)
assert before.count == 6

# Apply the reset pruning pass, which removes any redundant consecutive reset instructions
filter_pass = ExampleResetPruningPass()
filter_pass.on_module(mod)
print(filter_pass.module.ir())

# Count up the remaining number of reset instructions after transformation
after = ExampleInstructionCounterPass(
    lambda instr: "reset" in as_qis_gate(instr).get("gate", "")
)
after.on_module(filter_pass.module)
assert after.count == 3

# %%
