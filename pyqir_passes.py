# Licensed under the MIT License.

from pyqir import (
    BasicBlock,
    Constant,
    IntConstant,
    FloatConstant,
    Function,
    Instruction,
    Call,
    Module,
    Opcode,
    Phi,
    ICmp,
    Value,
    const,
    qubit,
    qubit_id,
    result,
    result_id,
    is_entry_point,
    add_string_attribute,
    SimpleModule,
)
from collections import OrderedDict
from typing import List, Dict, Callable


def as_qis_gate(instr: Instruction) -> Dict:
    if isinstance(instr, Call) and instr.callee.name.startswith("__quantum__qis__"):
        parts = instr.callee.name.split("__")
        return {
            "gate": parts[3] + ("_adj" if parts[4] == "adj" else ""),
            "qubit_args": [
                qubit_id(arg) for arg in instr.args if qubit_id(arg) is not None
            ],
            "result_args": [
                result_id(arg) for arg in instr.args if result_id(arg) is not None
            ],
            "other_args": [
                arg
                for arg in instr.args
                if qubit_id(arg) is None and result_id(arg) is None
            ],
        }
    return {}


class SimplePass:
    # A simple pass that visits each instruction in a module, function, or basic block. Subclasses can override the
    # appropriate methods to implement custom behavior. The default implementations do nothing.
    # This base class is a good choice for analysis passes that do not need to modify the module.

    def __init__(self):
        pass

    def on_module(self, module: Module | SimpleModule):
        if isinstance(module, SimpleModule):
            module = module._module
        # Visit the non-entry functions first. This will include any function declarations.
        for func in filter(lambda f: not is_entry_point(f), module.functions):
            self.on_function(func)
        # Visit the entry function(s)
        for func in filter(is_entry_point, module.functions):
            self.on_function(func)

    def on_function(self, function: Function):
        # Visit each basic block in the function. The entry block is visited first, follow by other blocks
        # in order determined by the topology of the control flow graph.
        blocks: List[BasicBlock] = []
        if len(function.basic_blocks):
            blocks.append(function.basic_blocks[0])
        to_visit = OrderedDict()
        for block in blocks:
            to_visit[block.name] = block
            if block.terminator is not None:
                for b in reversed(block.terminator.successors):
                    if b.name not in to_visit:
                        blocks.append(b)
        for block in to_visit.values():
            self.on_block(block)

    def on_block(self, block: BasicBlock):
        # Visit each instruction in the block, including the terminator, in order.
        for instr in block.instructions:
            self.on_instruction(instr)

    def on_instruction(self, instruction: Instruction):
        # Dispatch to the appropriate method based on the instruction's opcode, if we have one.
        opcode = instruction.opcode
        if opcode == Opcode.AND:
            self.on_and_instr(instruction)
        elif opcode == Opcode.OR:
            self.on_or_instr(instruction)
        elif opcode == Opcode.XOR:
            self.on_xor_instr(instruction)
        elif opcode == Opcode.ADD:
            self.on_add_instr(instruction)
        elif opcode == Opcode.SUB:
            self.on_sub_instr(instruction)
        elif opcode == Opcode.MUL:
            self.on_mul_instr(instruction)
        elif opcode == Opcode.SDIV:
            self.on_div_instr(instruction)
        elif opcode == Opcode.SHL:
            self.on_shl_instr(instruction)
        elif opcode == Opcode.LSHR:
            self.on_lshr_instr(instruction)
        elif opcode == Opcode.ICMP:
            self.on_icmp_instr(instruction)
        elif opcode == Opcode.CALL:
            self.on_call_instr(instruction)
        elif opcode == Opcode.BR and len(instruction.operands) == 1:
            self.on_br_instr(instruction)
        elif opcode == Opcode.BR and len(instruction.operands) > 1:
            self.on_condbr_instr(instruction)
        elif opcode == Opcode.PHI:
            self.on_phi_instr(instruction)
        elif opcode == Opcode.RET:
            self.on_ret_instr(instruction)
        elif opcode == Opcode.ZEXT:
            self.on_zext_instr(instruction)
        elif opcode == Opcode.TRUNC:
            self.on_trunc_instr(instruction)
        else:
            self.on_other_instr(instruction)

    def on_and_instr(self, instr: Instruction):
        pass

    def on_or_instr(self, instr: Instruction):
        pass

    def on_xor_instr(self, instr: Instruction):
        pass

    def on_add_instr(self, instr: Instruction):
        pass

    def on_sub_instr(self, instr: Instruction):
        pass

    def on_mul_instr(self, instr: Instruction):
        pass

    def on_div_instr(self, instr: Instruction):
        pass

    def on_shl_instr(self, instr: Instruction):
        pass

    def on_lshr_instr(self, instr: Instruction):
        pass

    def on_icmp_instr(self, instr: ICmp):
        pass

    def on_call_instr(self, instr: Call):
        gate = as_qis_gate(instr)
        if len(gate) > 0:
            # Handle the Q# canonical gate set.
            if gate["gate"] == "ccx":
                self.on_cxx_gate(instr, gate)
            elif gate["gate"] == "cx":
                self.on_cx_gate(instr, gate)
            elif gate["gate"] == "cy":
                self.on_cy_gate(instr, gate)
            elif gate["gate"] == "cz":
                self.on_cz_gate(instr, gate)
            elif gate["gate"] == "rx":
                self.on_rx_gate(instr, gate)
            elif gate["gate"] == "rxx":
                self.on_rxx_gate(instr, gate)
            elif gate["gate"] == "ry":
                self.on_ry_gate(instr, gate)
            elif gate["gate"] == "ryy":
                self.on_ryy_gate(instr, gate)
            elif gate["gate"] == "rz":
                self.on_rz_gate(instr, gate)
            elif gate["gate"] == "rzz":
                self.on_rzz_gate(instr, gate)
            elif gate["gate"] == "h":
                self.on_h_gate(instr, gate)
            elif gate["gate"] == "s":
                self.on_s_gate(instr, gate)
            elif gate["gate"] == "s_adj":
                self.on_sadj_gate(instr, gate)
            elif gate["gate"] == "swap":
                self.on_swap_gate(instr, gate)
            elif gate["gate"] == "t":
                self.on_t_gate(instr, gate)
            elif gate["gate"] == "t_adj":
                self.on_tadj_gate(instr, gate)
            elif gate["gate"] == "x":
                self.on_x_gate(instr, gate)
            elif gate["gate"] == "y":
                self.on_y_gate(instr, gate)
            elif gate["gate"] == "z":
                self.on_z_gate(instr, gate)
            elif gate["gate"] == "m":
                self.on_m_gate(instr, gate)
            elif gate["gate"] == "mresetz":
                self.on_mresetz_gate(instr, gate)
            elif gate["gate"] == "reset":
                self.on_reset_gate(instr, gate)
            else:
                self.on_other_gate(instr, gate)

    def on_br_instr(self, instr: Instruction):
        pass

    def on_condbr_instr(self, instr: Instruction):
        pass

    def on_phi_instr(self, instr: Phi):
        pass

    def on_ret_instr(self, instr: Instruction):
        pass

    def on_zext_instr(self, instr: Instruction):
        pass

    def on_trunc_instr(self, instr: Instruction):
        pass

    def on_other_instr(self, instr: Instruction):
        pass

    def on_cxx_gate(self, instr: Call, gate: Dict):
        pass

    def on_cx_gate(self, instr: Call, gate: Dict):
        pass

    def on_cy_gate(self, instr: Call, gate: Dict):
        pass

    def on_cz_gate(self, instr: Call, gate: Dict):
        pass

    def on_rx_gate(self, instr: Call, gate: Dict):
        pass

    def on_rxx_gate(self, instr: Call, gate: Dict):
        pass

    def on_ry_gate(self, instr: Call, gate: Dict):
        pass

    def on_ryy_gate(self, instr: Call, gate: Dict):
        pass

    def on_rz_gate(self, instr: Call, gate: Dict):
        pass

    def on_rzz_gate(self, instr: Call, gate: Dict):
        pass

    def on_h_gate(self, instr: Call, gate: Dict):
        pass

    def on_s_gate(self, instr: Call, gate: Dict):
        pass

    def on_sadj_gate(self, instr: Call, gate: Dict):
        pass

    def on_swap_gate(self, instr: Call, gate: Dict):
        pass

    def on_t_gate(self, instr: Call, gate: Dict):
        pass

    def on_tadj_gate(self, instr: Call, gate: Dict):
        pass

    def on_x_gate(self, instr: Call, gate: Dict):
        pass

    def on_y_gate(self, instr: Call, gate: Dict):
        pass

    def on_z_gate(self, instr: Call, gate: Dict):
        pass

    def on_m_gate(self, instr: Call, gate: Dict):
        pass

    def on_mresetz_gate(self, instr: Call, gate: Dict):
        pass

    def on_reset_gate(self, instr: Call, gate: Dict):
        pass

    def on_other_gate(self, instr: Call, gate: Dict):
        pass


class SimpleClonerPass(SimplePass):
    # A simple pass that clones a module. This is useful for transforming a module while keeping the original intact.
    # The pass will create a new `SimpleModule` with the same name and context as the original, and will clone the functions
    # basic blocks, instructions, and values. Note that the pass does not clone the module metadata of the
    # original module. Only the content that is compatible with a `SimpleModule` is cloned.
    # This base class is a good choice for transformation passes that need to modify the module.

    def __init__(self):
        super().__init__()
        self.module = None
        self.values: Dict[Value, Value] = {}

    def _create_constant(self, value: Constant):
        if isinstance(value, IntConstant) or isinstance(value, FloatConstant):
            self.values[value] = const(value.type, value.value)
        elif qubit_id(value) is not None:
            self.values[value] = qubit(self.module.context, qubit_id(value))
        elif result_id(value) is not None:
            self.values[value] = result(self.module.context, result_id(value))
        elif value.is_null:
            self.values[value] = Constant.null(value.type)
        else:
            raise ValueError(f"Unhandled constant: {value}")
        return self.values[value]

    def _map_val(self, value: Value) -> Value:
        mapped = self.values.get(value, None)
        if mapped is None:
            if isinstance(value, Constant):
                mapped = self._create_constant(value)
                self.values[value] = mapped
            else:
                raise ValueError(f"Unexpected non-constant value: {value}")
        return mapped

    def _clone_call(self, instr: Call):
        res = self.module.builder.call(
            self._map_val(instr.callee),
            [self._map_val(arg) for arg in instr.args],
        )
        if not (isinstance(res, Constant) and res.is_null()):
            self.values[instr] = res

    def on_module(self, module: Module | SimpleModule):
        if isinstance(module, SimpleModule):
            module = module._module
        entry_func = next(filter(is_entry_point, module.functions))
        num_qubits = int(entry_func.attributes.func["required_num_qubits"].string_value)
        num_results = int(
            entry_func.attributes.func["required_num_results"].string_value
        )
        self.module = SimpleModule(
            module.source_filename,
            context=module.context,
            num_qubits=num_qubits,
            num_results=num_results,
        )
        super().on_module(module)
        self.module._module.verify()

    def on_function(self, function: Function):
        # If this is not the entry function, we need to add it to the module.
        if not is_entry_point(function):
            self.values[function] = self.module.add_external_function(
                function.name, function.type
            )
        else:
            self.values[function] = self.module.entry_point
        # Copy the function's attributes.
        for attr in function.attributes.func:
            add_string_attribute(
                self.values[function], attr.string_kind, attr.string_value
            )
        self.current_function = function
        if len(function.basic_blocks) > 0:
            self.values[function.basic_blocks[0]] = self.module.entry_block
            for block in function.basic_blocks[1:]:
                self.values[block] = BasicBlock(
                    self.module.context, block.name, self.values[function]
                )
        super().on_function(function)

    def on_block(self, block: BasicBlock):
        self.module.builder.insert_at_end(self.values[block])
        super().on_block(block)

    def on_and_instr(self, instr: Instruction):
        self.values[instr] = self.module.builder.and_(
            self._map_val(instr.operands[0]),
            self._map_val(instr.operands[1]),
        )
        super().on_and_instr(instr)

    def on_or_instr(self, instr: Instruction):
        self.values[instr] = self.module.builder.or_(
            self._map_val(instr.operands[0]),
            self._map_val(instr.operands[1]),
        )
        super().on_or_instr(instr)

    def on_xor_instr(self, instr: Instruction):
        self.values[instr] = self.module.builder.xor(
            self._map_val(instr.operands[0]),
            self._map_val(instr.operands[1]),
        )
        super().on_xor_instr(instr)

    def on_add_instr(self, instr: Instruction):
        self.values[instr] = self.module.builder.add(
            self._map_val(instr.operands[0]),
            self._map_val(instr.operands[1]),
        )
        super().on_add_instr(instr)

    def on_sub_instr(self, instr: Instruction):
        self.values[instr] = self.module.builder.sub(
            self._map_val(instr.operands[0]),
            self._map_val(instr.operands[1]),
        )
        super().on_sub_instr(instr)

    def on_mul_instr(self, instr: Instruction):
        self.values[instr] = self.module.builder.mul(
            self._map_val(instr.operands[0]),
            self._map_val(instr.operands[1]),
        )
        super().on_mul_instr(instr)

    def on_div_instr(self, instr: Instruction):
        # self.values[instr] = self.module.builder.sdiv(
        #     self.get_or_create_value(instr.operands[0]),
        #     self.get_or_create_value(instr.operands[1]),
        # )
        # super().on_div_instr(instr)
        raise NotImplemented("sdiv not supported in builder")

    def on_shl_instr(self, instr: Instruction):
        self.values[instr] = self.module.builder.shl(
            self._map_val(instr.operands[0]),
            self._map_val(instr.operands[1]),
        )
        super().on_shl_instr(instr)

    def on_lshr_instr(self, instr: Instruction):
        self.values[instr] = self.module.builder.lshr(
            self._map_val(instr.operands[0]),
            self._map_val(instr.operands[1]),
        )
        super().on_lshr_instr(instr)

    def on_icmp_instr(self, instr: ICmp):
        self.values[instr] = self.module.builder.icmp(
            instr.predicate,
            self._map_val(instr.operands[0]),
            self._map_val(instr.operands[1]),
        )
        super().on_icmp_instr(instr)

    def on_call_instr(self, instr: Call):
        gate = as_qis_gate(instr)
        if len(gate) > 0:
            # Handle the Q# canonical gate set.
            if gate["gate"] == "ccx":
                self.on_cxx_gate(instr, gate)
            elif gate["gate"] == "cx":
                self.on_cx_gate(instr, gate)
            elif gate["gate"] == "cy":
                self.on_cy_gate(instr, gate)
            elif gate["gate"] == "cz":
                self.on_cz_gate(instr, gate)
            elif gate["gate"] == "rx":
                self.on_rx_gate(instr, gate)
            elif gate["gate"] == "rxx":
                self.on_rxx_gate(instr, gate)
            elif gate["gate"] == "ry":
                self.on_ry_gate(instr, gate)
            elif gate["gate"] == "ryy":
                self.on_ryy_gate(instr, gate)
            elif gate["gate"] == "rz":
                self.on_rz_gate(instr, gate)
            elif gate["gate"] == "rzz":
                self.on_rzz_gate(instr, gate)
            elif gate["gate"] == "h":
                self.on_h_gate(instr, gate)
            elif gate["gate"] == "s":
                self.on_s_gate(instr, gate)
            elif gate["gate"] == "s_adj":
                self.on_sadj_gate(instr, gate)
            elif gate["gate"] == "swap":
                self.on_swap_gate(instr, gate)
            elif gate["gate"] == "t":
                self.on_t_gate(instr, gate)
            elif gate["gate"] == "t_adj":
                self.on_tadj_gate(instr, gate)
            elif gate["gate"] == "x":
                self.on_x_gate(instr, gate)
            elif gate["gate"] == "y":
                self.on_y_gate(instr, gate)
            elif gate["gate"] == "z":
                self.on_z_gate(instr, gate)
            elif gate["gate"] == "m":
                self.on_m_gate(instr, gate)
            elif gate["gate"] == "mresetz":
                self.on_mresetz_gate(instr, gate)
            elif gate["gate"] == "reset":
                self.on_reset_gate(instr, gate)
            else:
                self.on_other_gate(instr, gate)
        else:
            self._clone_call(instr)

    def on_br_instr(self, instr: Instruction):
        self.values[instr] = self.module.builder.br(self._map_val(instr.operands[0]))
        super().on_br_instr(instr)

    def on_condbr_instr(self, instr: Instruction):
        self.values[instr] = self.module.builder.condbr(
            self._map_val(instr.operands[0]),
            self._map_val(instr.operands[1]),
            self._map_val(instr.operands[2]),
        )
        super().on_condbr_instr(instr)

    def on_phi_instr(self, instr: Phi):
        phi = self.module.builder.phi(instr.type)
        self.values[instr] = phi
        for val, block in instr.incoming:
            phi.add_incoming(self._map_val(val), self._map_val(block))
        super().on_phi_instr(instr)

    def on_ret_instr(self, instr: Instruction):
        # A `ret void` is automatically added by simple builder, so no need to add it here.
        if len(instr.operands) > 0:
            # The input function has a non-void return, which can't be handled.
            raise NotImplemented("return with non-void argument")
        super().on_ret_instr(instr)

    def on_zext_instr(self, instr: Instruction):
        self.values[instr] = self.module.builder.zext(
            self._map_val(instr.operands[0]), instr.type
        )
        super().on_zext_instr(instr)

    def on_trunc_instr(self, instr: Instruction):
        self.values[instr] = self.module.builder.trunc(
            self._map_val(instr.operands[0]), instr.type
        )
        super().on_trunc_instr(instr)

    def on_other_instr(self, instr: Instruction):
        raise NotImplementedError(f"Unhandled instruction: {instr}")

    def on_cxx_gate(self, instr: Call, gate: Dict):
        self._clone_call(instr)

    def on_cx_gate(self, instr: Call, gate: Dict):
        self._clone_call(instr)

    def on_cy_gate(self, instr: Call, gate: Dict):
        self._clone_call(instr)

    def on_cz_gate(self, instr: Call, gate: Dict):
        self._clone_call(instr)

    def on_rx_gate(self, instr: Call, gate: Dict):
        self._clone_call(instr)

    def on_rxx_gate(self, instr: Call, gate: Dict):
        self._clone_call(instr)

    def on_ry_gate(self, instr: Call, gate: Dict):
        self._clone_call(instr)

    def on_ryy_gate(self, instr: Call, gate: Dict):
        self._clone_call(instr)

    def on_rz_gate(self, instr: Call, gate: Dict):
        self._clone_call(instr)

    def on_rzz_gate(self, instr: Call, gate: Dict):
        self._clone_call(instr)

    def on_h_gate(self, instr: Call, gate: Dict):
        self._clone_call(instr)

    def on_s_gate(self, instr: Call, gate: Dict):
        self._clone_call(instr)

    def on_sadj_gate(self, instr: Call, gate: Dict):
        self._clone_call(instr)

    def on_swap_gate(self, instr: Call, gate: Dict):
        self._clone_call(instr)

    def on_t_gate(self, instr: Call, gate: Dict):
        self._clone_call(instr)

    def on_tadj_gate(self, instr: Call, gate: Dict):
        self._clone_call(instr)

    def on_x_gate(self, instr: Call, gate: Dict):
        self._clone_call(instr)

    def on_y_gate(self, instr: Call, gate: Dict):
        self._clone_call(instr)

    def on_z_gate(self, instr: Call, gate: Dict):
        self._clone_call(instr)

    def on_m_gate(self, instr: Call, gate: Dict):
        self._clone_call(instr)

    def on_mresetz_gate(self, instr: Call, gate: Dict):
        self._clone_call(instr)

    def on_reset_gate(self, instr: Call, gate: Dict):
        self._clone_call(instr)

    def on_other_gate(self, instr: Call, gate: Dict):
        self._clone_call(instr)


class ExampleInstructionCounterPass(SimplePass):
    # An example analysis pass that counts the number of instructions that satisfy a given predicate. The predicate is a callable
    # that takes an instruction and returns a boolean indicating whether the instruction should be counted.

    def __init__(self, counter: Callable[[Instruction], bool]):
        self.counter = counter
        self.count = 0
        super().__init__()

    def on_instruction(self, instr: Instruction):
        if self.counter(instr):
            self.count += 1


class ExampleInstrFilterPass(SimpleClonerPass):
    # An example transformation pass that filters instructions based on a predicate. The filter is applied to each instruction, and only
    # those that pass the filter are cloned into the new module. The filter is a callable that takes an instruction and
    # returns a boolean indicating whether the instruction should be included.

    def __init__(self, filter: Callable[[Instruction], bool]):
        super().__init__()
        self.filter = filter

    def on_instruction(self, instruction: Instruction):
        if self.filter(instruction):
            super().on_instruction(instruction)


class ExampleResetPruningPass(SimpleClonerPass):
    # An example transformation pass that prunes redundant reset instructions. If a qubit is reset and then reset again without being
    # used in a gate in between, the second reset is dropped.

    def __init__(self):
        super().__init__()

    def on_block(self, block: BasicBlock):
        self.is_reset: Dict[Value, bool] = {}
        super().on_block(block)

    def on_instruction(self, instruction: Instruction):
        gate = as_qis_gate(instruction)
        if gate.get("gate", None) == "mresetz":
            self.is_reset[gate["qubit_args"][0]] = True
        elif gate.get("gate", None) == "reset":
            if self.is_reset.get(gate["qubit_args"][0], False):
                # This reset is redundant, skip it to prune it.
                return
            self.is_reset[gate["qubit_args"][0]] = True
        elif len(gate.get("qubit_args", [])) > 0:
            # If a qubit is used in a gate, it is no longer reset.
            for q in gate["qubit_args"]:
                self.is_reset[q] = False
        super().on_instruction(instruction)


class ExampleLogicSeparationPass(SimpleClonerPass):
    # An example transformation pass that separates logic within a block into different sections based on the type of instruction. The sections are:
    # 1. Setup: Instructions that prepare classical values that do not depend on quantum state.
    # 2. Quantum gates: Instructions that apply quantum gates and measurements.
    # 3. Readout: Instructions that read out measurement results as bits.
    # 4. Processing: Instructions that perform classical processing dependent on measurement results.
    # 5. Output Recording: Instructions that record output values, if any.
    # The order of instructions in each of these sections is preserved. Note that the terminator is always placed at the end.
    # This pass does NOT handle blocks where classical values computed in step 4 (Processing) are then used in later quantum gates,
    # and may put such instructions in the wrong section, causing module validation failures.

    def __init__(self):
        super().__init__()

    def on_block(self, block: BasicBlock):
        self.module.builder.insert_at_end(self.values[block])
        setup: List[Instruction] = []
        gates: List[Instruction] = []
        readout: List[Instruction] = []
        processing: List[Instruction] = []
        output_recording: List[Instruction] = []
        # Process all the instructions except the terminator.
        for instr in block.instructions[:-1]:
            gate = as_qis_gate(instr)
            if len(gate) == 0:
                # Not a quantum gate, so must be setup, readout, or processing.
                # If any operand is from readout or processing, it is processing.
                if isinstance(instr, Call) and instr.callee.name.endswith(
                    "record_output"
                ):
                    output_recording.append(instr)
                elif any([op in readout or op in processing for op in instr.operands]):
                    processing.append(instr)
                else:
                    setup.append(instr)
            elif gate["gate"] == "read_result":
                readout.append(instr)
            else:
                gates.append(instr)
        # Process the instructions in order.
        for instr in setup + gates + readout + processing + output_recording:
            self.on_instruction(instr)
        # Process the terminator.
        self.on_instruction(block.instructions[-1])


class ExampleQubitShiftPass(SimpleClonerPass):
    # An example transformation pass that shifts qubit indices in quantum gates by a fixed amount. This is useful for transforming a module
    # that uses a different number of qubits than the original. The shift is applied to qubit arguments in quantum gates,
    # but not to result arguments or other arguments.

    def __init__(self, shift: int):
        super().__init__()
        self.shift = shift

    def _shift_qubit(self, qubit: Value) -> Value:
        if qubit_id(qubit) is not None:
            return qubit(self.module.context, qubit_id(qubit) + self.shift)
        return qubit

    def on_module(self, module: Module | SimpleModule):
        if isinstance(module, SimpleModule):
            module = module._module
        entry_func = next(filter(is_entry_point, module.functions))
        num_qubits = int(entry_func.attributes.func["required_num_qubits"].string_value)
        num_results = int(
            entry_func.attributes.func["required_num_results"].string_value
        )
        self.module = SimpleModule(
            module.source_filename,
            context=module.context,
            num_qubits=num_qubits + self.shift,
            num_results=num_results,
        )
        SimplePass.on_module(self, module)

    def on_call_instr(self, instr: Call):
        res = self.module.builder.call(
            self._map_val(instr.callee),
            [self._map_val(self._shift_qubit(arg)) for arg in instr.args],
        )
        if not (isinstance(res, Constant) and res.is_null()):
            self.values[instr] = res
