//===- QCSOps.td - System dialect ops ----------------------*- tablegen -*-===//
//
// (C) Copyright IBM 2023.
//
// This code is part of Qiskit.
//
// This code is licensed under the Apache License, Version 2.0 with LLVM
// Exceptions. You may obtain a copy of this license in the LICENSE.txt
// file in the root directory of this source tree.
//
// Any modifications or derivative works of this code must retain this
// copyright notice, and modified files need to carry a notice indicating
// that they have been altered from the originals.
//
//===----------------------------------------------------------------------===//
///
/// This is the main operation definition file for Quantum Control System
/// operations.
///
//===----------------------------------------------------------------------===//

#ifndef QCS_OPS
#define QCS_OPS

// TODO: Temporary, until constraints between `OpenQASM3`, `QUIR`, `Pulse`, and
// `System` dialects are ironed out.
include "Dialect/QUIR/IR/QUIRInterfaces.td"
include "Dialect/QUIR/IR/QUIRTraits.td"
include "Dialect/QUIR/IR/QUIRTypeConstraints.td"

include "Dialect/QCS/IR/QCSBase.td"

include "mlir/Dialect/LLVMIR/LLVMOpBase.td"
include "mlir/Interfaces/CallInterfaces.td"
include "mlir/Interfaces/ControlFlowInterfaces.td"
include "mlir/IR/FunctionInterfaces.td"
include "mlir/IR/OpAsmInterface.td"
include "mlir/IR/SymbolInterfaces.td"

// Define a side effect that identifies an operation as not dead while not
// interfering with memory operations (e.g., allows store-forwarding across
// this operation).
// Note that operations without memory effects defined will be treated
// conservatively (i.e., not making any assumptions).
// see lib/Interfaces/SideEffectInterfaces.cpp:isTriviallyDeadImpl()
// see lib/Dialect/Affine/Utils/Utils.cpp:hasNoInterveningEffect()
def NonInterferingNonDeadSideEffect : MemoryEffects<[MemFree<DefaultResource>]>;

//===----------------------------------------------------------------------===//
// SystemInitOp
//===----------------------------------------------------------------------===//

def QCS_SystemInitOp : QCS_Op<"init", [IsolatedFromAbove]> {
    let summary = "Initializes the system";
    let description = [{
        The `qcs.system_init` operation causes the system to be initialized,
        preparing for execution and synchronizing the system.
    }];
    let assemblyFormat = [{
        attr-dict
    }];
}

//===----------------------------------------------------------------------===//
// SystemFinalizeOp
//===----------------------------------------------------------------------===//

def QCS_SystemFinalizeOp : QCS_Op<"finalize", [IsolatedFromAbove]> {
    let summary = "Finalizes the system";
    let description = [{
        The `qcs.system_finalize` operation causes the system to be finalized,
        cleaning up results and shutting down execution.
    }];
    let assemblyFormat = [{
        attr-dict
    }];
}

//===----------------------------------------------------------------------===//
// SynchronizeOp
//===----------------------------------------------------------------------===//

def QCS_SynchronizeOp : QCS_Op<"synchronize", [
            DeclareOpInterfaceMethods<QubitOpInterface, ["getOperatedQubits"]>,
            NonInterferingNonDeadSideEffect]> {
    let summary = "Synchronize the system";
    let description = [{
        The `qcs.synchronize` operation causes the system to perform a
        synchronization so that each of the independent controllers is locked
        into the same time step again. The synchronize op accepts qubit
        arguments, this is to simplify the tracking of synchronization timing
        relative to other qubit-based operations, like measurement and reset.
    }];
    let arguments = (ins Variadic<AnyQubit>:$qubits);
    let results = (outs);

    let assemblyFormat = [{
        $qubits attr-dict `:` functional-type($qubits, results)
    }];
}

//===----------------------------------------------------------------------===//
// BroadcastOp
//===----------------------------------------------------------------------===//

def QCS_BroadcastOp : QCS_Op<"broadcast", [NonInterferingNonDeadSideEffect]> {
    let summary = "Broadcast a value from this controller to all others";
    let description = [{
        The `qcs.broadcast` operation represents a broadcast command that sends a value
        from this controller to all others. All other controllers should have a
        corresponding `qcs.recv` operation. If only one controller should receive the
        message then the `qcs.send` operation should be used instead.

        Example:
        ```mlir
        %angle1 = quir.constant #quir.angle<0.2> : !quir.angle<20>
        qcs.broadcast %angle1 : !quir.angle<20>
        ```
    }];

    let arguments = (ins AnyClassical:$val);

    let assemblyFormat = [{
        attr-dict $val `:` type($val)
    }];
}

//===----------------------------------------------------------------------===//
// DelayCyclesOp
//===----------------------------------------------------------------------===//

// TODO: Should this be a "unitary op" at the system level?
def QCS_DelayCyclesOp : QCS_Op<"delay_cycles", [
            UnitaryOp,
            DeclareOpInterfaceMethods<QubitOpInterface, ["getOperatedQubits"]>,
            NonInterferingNonDeadSideEffect]> {
    let summary = "Add a delay operation of a given number of cycles to a qubit";
    let description = [{
        The `qcs.delay_cycles` operation represents a delay operation of the given
        number of cycles to a qubit, group of qubits, or all qubits (when no
        target qubit is given).

        Example:
        ```mlir
        "qcs.delay_cycles"(%q1_1) { time = 1000 : i64}: (!quir.qubit) -> ()
        ```
    }];

    let arguments = (ins I64Attr:$time, Variadic<AnyQubit>:$qubits);
    let results = (outs );

    let assemblyFormat = [{
        `(` $qubits `)` attr-dict `:` functional-type($qubits, results)
    }];
}

def IndexArrayAttr : TypedArrayAttrBase<IndexAttr,
                                        "index array attribute"> {
  let constBuilderCall = "$_builder.getIndexArrayAttr($0)";
}

//===----------------------------------------------------------------------===//
// SendOp
//===----------------------------------------------------------------------===//

def QCS_SendOp : QCS_Op<"send", [NonInterferingNonDeadSideEffect]> {
    let summary = "Send a classical value from this controller to another";
    let description = [{
        The `qcs.send` operation represents a send command from one controller to another.
        A corresponding `qcs.recv` operation should receive the information sent.

        Example:
        ```mlir
        %cbit = "quir.measure"(%target) : (!quir.qubit<1>) -> i1
        qcs.send %cbit : i1
        ```
    }];

    let arguments = (ins AnyClassical:$val, IndexAttr:$id);

    let assemblyFormat = [{
        attr-dict $val `to` $id `:` type($val)
    }];
}

//===----------------------------------------------------------------------===//
// RecvOp
//===----------------------------------------------------------------------===//

def QCS_RecvOp : QCS_Op<"recv", [NonInterferingNonDeadSideEffect]> {
    let summary = "Receive classical values from other controllers";
    let description = [{
        The `qcs.recv` operation represents a receive command for potentially
        multiple values. The fromIds array attribute indicates the Id of the
        sending control node for each independent value.

        Example:
        ```mlir
        %meas:2 = qcs.recv {fromId = [0 : index, 1 : index]} : i1, i1
        ```
    }];

    let arguments = (ins OptionalAttr<IndexArrayAttr>:$fromIds);
    let results = (outs Variadic<AnyClassical>:$vals);

    let assemblyFormat = [{
        attr-dict `:` type($vals)
    }];
}

//===----------------------------------------------------------------------===//
// ParallelControlFlowOp
//===----------------------------------------------------------------------===//

def QCS_ParallelControlFlowOp : QCS_Op<"parallel_control_flow", [
                        SingleBlockImplicitTerminator<"ParallelEndOp">,
                        RecursiveMemoryEffects]> {
    let summary = "Contain a group of control flow ops that can run in parallel";
    let description = [{
        The `qcs.parallel_control_flow` operation has a single region and block
        that holds a collection of scf control flow ops that can be executed in
        parallel.

        Example:
        ```mlir
        qcs.parallel_control_flow {
            scf.if (%c0) {
                quir.gate_func.call @x(%q0) : (!quir.qubit<1>) -> ()
            } {quir.physicalIds = [0 : i32]}
            scf.if (%c1) {
                quir.gate_func.call @x(%q1) : (!quir.qubit<1>) -> ()
            } {quir.physicalIds = [1 : i32]}
        }
        ```
    }];

    let regions = (region SizedRegion<1>:$region);

    let assemblyFormat = "$region attr-dict";
}

//===----------------------------------------------------------------------===//
// ParallelEndOp
//===----------------------------------------------------------------------===//

def QCS_ParallelEndOp : QCS_Op<"parallel_control_flow_end", [
    Terminator, HasParent<"ParallelControlFlowOp">, Pure]> {
  let summary = "A pseudo-op that marks the end of a `qcs.parallel_control_flow` op.";
  let description = [{
    This op terminates the only block inside the only region of a
    `qcs.parallel_control_flow` op.
  }];

  let hasCustomAssemblyFormat = 1;
}

//===----------------------------------------------------------------------===//
// ShotInitOp
//===----------------------------------------------------------------------===//

def QCS_ShotInitOp : QCS_Op<"shot_init", [IsolatedFromAbove]> {
    let summary = "Initialization for shots";
    let description = [{
        The `qcs.shot_init` operation causes the system to initialize a shot to ensure
        consistent initial conditions for each execution of a given quantum algorithm.
    }];
    let assemblyFormat = [{
        attr-dict
    }];
}

//===----------------------------------------------------------------------===//
// DeclareParameterOp
//===----------------------------------------------------------------------===//

def QCS_DeclareParameterOp : QCS_Op<"declare_parameter", [Symbol]> {
    let summary = "system input parameter subject to post compilation updates";
    let description = [{
        The `qcs.declare_parameter` operation adds a symbol defining an input parameter
        which may be modified after compilation before/during program invocation.
        The value of the input parameter
        may be obtained using the qcs.use_input_parameter operation.

        Example:

        ```
        // quir.angle input parameter
        qcs.declare_parameter  "theta" : !quir.angle<64> = 3.14159
        ```
    }];

    let arguments = (ins
        SymbolNameAttr:$sym_name,
        TypeAttr:$type,
        OptionalAttr<AnyAttr>:$initial_value
    );

    let results = (outs);

    let assemblyFormat = [{
        attr-dict $sym_name `:` $type (`=` $initial_value^)?
    }];

    let builders = [
        OpBuilder<(ins "::llvm::StringRef":$sym_name, "::mlir::TypeAttr":$type), [{
            $_state.addAttribute("sym_name", $_builder.getStringAttr(sym_name));
            $_state.addAttribute("type", type);
        }]>,
        OpBuilder<(ins "::llvm::StringRef":$sym_name, "::mlir::TypeAttr":$type, "Attribute":$value), [{
            $_state.addAttribute("sym_name", $_builder.getStringAttr(sym_name));
            $_state.addAttribute("type", type);
            $_state.addAttribute("initial_value", value);
        }]>,
    ];
}

//===----------------------------------------------------------------------===//
// ParameterLoadOp
//===----------------------------------------------------------------------===//

def QCS_ParameterLoadOp : QCS_Op<"parameter_load",
                        [DeclareOpInterfaceMethods<SymbolUserOpInterface>]> {
    let summary = "Use the current value of a parameter";
    let description = [{
        The operation `qcs.parameter_load` returns the current value of the
        classical parameter with the given name.

        Example:

        ```mlir
        %2 = qcs.parameter_load "a" : !quir.angle<64>
        ```
    }];

    let arguments = (ins
        FlatSymbolRefAttr:$parameter_name
    );

    let results = (outs AnyClassical:$res);

    let extraClassDeclaration = [{
        // Return the initial value - using ParameterInitialValueAnalysis
        ParameterType getInitialValue(llvm::StringMap<ParameterType> &parameterNames);

        // Return the initial value - slower SymbolTable version
        ParameterType getInitialValue();
    }];

    let assemblyFormat = [{
        $parameter_name `:` type($res) attr-dict
    }];
}

//===----------------------------------------------------------------------===//
// ConstantOp
//===----------------------------------------------------------------------===//

def QCS_ConstantOp : QCS_Op<"constant",
    [ConstantLike, Pure,
     DeclareOpInterfaceMethods<OpAsmOpInterface, ["getAsmResultNames"]>]> {
  let summary = "constant";
  let description = [{
    The `qcs.constant` operation produces an SSA value from a symbol reference
    to a `qcs.remote_procedure` operation

    Example:

    ```mlir
    // Reference to remote procedure @myfn.
    %2 = qcs.constant @myprocedure : (f32) -> f32

    // Equivalent generic forms
    %2 = "qcs.constant"() { value = @myprocedure } : () -> (f32) -> f32
    ```

    MLIR does not allow direct references to functions in SSA operands because
    the compiler is multithreaded, and disallowing SSA values to directly
    reference a function simplifies this.
  }];

  let arguments = (ins FlatSymbolRefAttr:$value);
  let results = (outs AnyType);
  let assemblyFormat = "attr-dict $value `:` type(results)";

  let extraClassDeclaration = [{
    /// Returns true if a constant operation can be built with the given value
    /// and result type.
    static bool isBuildableWith(Attribute value, Type type);
  }];

  let hasFolder = 1;
  let hasVerifier = 1;
}

//===----------------------------------------------------------------------===//
// SendRemoteProcedureCallOp
//===----------------------------------------------------------------------===//

def QCS_SendRemoteProcedureCallOp : QCS_Op<"send_rpc", [MemRefsNormalizable]> {
    let summary = "Send a remote procedure call to another component in the system.";

    let description = [{
        This op sends a remote procedure call to another component in the system by idenfitier.

        Example:
        ```mlir
        ^bb2:
        %2 = call @someFn()
        qcf.send_rpc
        ```
    }];

    let arguments = (ins FlatSymbolRefAttr:$callee, Variadic<AnyType>:$operands);
    let results = (outs Variadic<AnyType>);

    let assemblyFormat = [{
        $callee `(` $operands `)` attr-dict `:` functional-type($operands, results)
    }];
}

//===----------------------------------------------------------------------===//
// AwaitRemoteProcedureCallOp
//===----------------------------------------------------------------------===//

def QCS_AwaitRemoteProcedureCallOp : QCS_Op<"await_rpc", [
    Terminator, Pure]> {
    let summary = "A pseudo-op that signifies the end of a block (procedure) for which the next block (procedure) will be invoked remotely.";

    let description = [{
        This op terminates the block and signifies the need to receive and execute a remote block
        call handing over flow control to another component of the system

        Example:
        ```mlir
        ^bb2:
        %2 = call @someFn()
        qcf.await_rpc
        ```
    }];

    let assemblyFormat = [{
            attr-dict
        }];
}

//===----------------------------------------------------------------------===//
// ExecuteRemoteProcedureCallOp
//===----------------------------------------------------------------------===//

def QCS_ExecuteRemoteProcedureCallOp : QCS_Op<"execute_rpc", [CallOpInterface, MemRefsNormalizable, DeclareOpInterfaceMethods<SymbolUserOpInterface>]> {
    let summary = "Execute a remote procedure call";
    let description = [{
        The `qcs.execute_rpc` operation represents the launching of a remote procedure in the target system.
        The callee function is attached to a symbol reference via an attribute. The arguments list must match
        the argument types provided by the callee. All classical input values to the sequence
        should be transferred before the invocation of the sequence routine to enable deterministic and
        real-time execution in hardware.

        Example:
        ```mlir
        qcs.remote_procedure @sequence0() -> i1 {
            %23 = pulse.call_sequence @circuit_0_sequence() {pulse.duration = 8264 : i64, pulse.timepoint = 8264 : i64} : (!pulse.mixed_frame, !pulse.waveform, !pulse.mixed_frame, !pulse.waveform, !pulse.mixed_frame) -> i1
            qcs.return %23
        }

        func.func @main() {
            %classical_result = pulse.execute_rpc @sequence0() : () -> (i1)
        }
        ```
    }];

    let arguments = (ins FlatSymbolRefAttr:$callee, Variadic<AnyClassical>:$operands);
    let results = (outs Variadic<AnyClassical>:$res);

    let assemblyFormat = [{
        $callee `(` $operands `)` attr-dict `:` functional-type($operands, results)
    }];

    let builders = [
        OpBuilder<(ins "RemoteProcedureOp":$callee, CArg<"ValueRange", "{}">:$operands), [{
            $_state.addOperands(operands);
            $_state.addAttribute("callee",SymbolRefAttr::get(callee));
            $_state.addTypes(callee.getFunctionType().getResults());
        }]>,
        OpBuilder<(ins "SymbolRefAttr":$callee, "TypeRange":$res,
        CArg<"ValueRange", "{}">:$operands), [{
            $_state.addOperands(operands);
            $_state.addAttribute("callee", callee);
            $_state.addTypes(res);
        }]>,
        OpBuilder<(ins "StringRef":$callee, "TypeRange":$res,
        CArg<"ValueRange", "{}">:$operands), [{
            build($_builder, $_state, SymbolRefAttr::get($_builder.getContext(), callee), res,
                    operands);
        }]>];

    let extraClassDeclaration = [{
        FunctionType getCalleeType();

        /// Get the argument operands to the called function.
        operand_range getArgOperands() {
            return {arg_operand_begin(), arg_operand_end()};
        }

        MutableOperandRange getArgOperandsMutable() {
            return getOperandsMutable();
        }

        operand_iterator arg_operand_begin() { return operand_begin(); }
        operand_iterator arg_operand_end() { return operand_end(); }

        /// Return the callee of this operation.
        CallInterfaceCallable getCallableForCallee() {
            return (*this)->getAttrOfType<SymbolRefAttr>("callee");
        }

        /// Set the callee for this operation.
        void setCalleeFromCallable(CallInterfaceCallable callee) {
            (*this)->setAttr("callee", callee.get<SymbolRefAttr>());
        }
    }];
}

//===----------------------------------------------------------------------===//
// ExecuteRemoteProcedureCallIndirectOp
//===----------------------------------------------------------------------===//

def QCS_ExecuteRemoteProcedureCallIndirectOp : QCS_Op<"execute_rpc_indirect", [
      CallOpInterface,
      TypesMatchWith<"callee input types match argument types",
                     "callee", "callee_operands",
                     "::llvm::cast<FunctionType>($_self).getInputs()">,
      TypesMatchWith<"callee result types match result types",
                     "callee", "results",
                     "::llvm::cast<FunctionType>($_self).getResults()">
    ]> {
  let summary = "indirect execute remote procedure call operation";
  let description = [{
    The `qcs.execute_rpc_indirect` operation represents an indirect call to a value
    of a remote procedure type. The operands and result types of the call must match the
    specified remote procedure type.

    Function values can be created with the
    [`func.constant` operation](#funcconstant-constantop).

    Example:

    ```mlir
    %func = qcs.constant @my_func : (i32) -> i32
    %result = qcs.execute_rpc_indirect %func(%0) : (i32) -> i32
    ```
  }];

  let arguments = (ins FunctionType:$callee,
                       Variadic<AnyClassical>:$callee_operands);
  let results = (outs Variadic<AnyClassical>:$results);

  let builders = [
    OpBuilder<(ins "Value":$callee, CArg<"ValueRange", "{}">:$operands), [{
      $_state.operands.push_back(callee);
      $_state.addOperands(operands);
      $_state.addTypes(::llvm::cast<FunctionType>(callee.getType()).getResults());
    }]>];

  let extraClassDeclaration = [{
    // TODO: Remove once migrated callers.
    ValueRange operands() { return getCalleeOperands(); }

    /// Get the argument operands to the called function.
    operand_range getArgOperands() {
      return {arg_operand_begin(), arg_operand_end()};
    }

    MutableOperandRange getArgOperandsMutable() {
      return getCalleeOperandsMutable();
    }

    operand_iterator arg_operand_begin() { return ++operand_begin(); }
    operand_iterator arg_operand_end() { return operand_end(); }

    /// Return the callee of this operation.
    CallInterfaceCallable getCallableForCallee() { return getCallee(); }

    /// Set the callee for this operation.
    void setCalleeFromCallable(CallInterfaceCallable callee) {
      setOperand(0, callee.get<Value>());
    }
  }];

  let hasCanonicalizeMethod = 1;
  let assemblyFormat = [{
    $callee `(` $callee_operands `)` attr-dict `:` type($callee)
  }];
}

//===----------------------------------------------------------------------===//
// RemoteProcedureOp
//===----------------------------------------------------------------------===//

def QCS_RemoteProcedureOp : QCS_Op<"remote_procedure", [
  AutomaticAllocationScope, CallableOpInterface,
  FunctionOpInterface, Symbol
]> {
  let summary = "An operation with a name containing a single `SSACFG` region corresponding to a remote procedure sequence";
  let description = [{

    Defines a remote sequence(function) operation representing a
    deterministically timed pulse sequence execution
    on the QPU *without* any classical compute or control-flow.
    Effectively corresponding to a callable "pulse sequence block".

    The RemoteProcedureOp is modelled after the FuncOp of the MLIR func
    dialect.

    A sequence may only contain operations legally allowed by the pulse
    dialect. All pulse-type operations should in-turn exist within regions
    contained by `pulse.sequence` operations.

    All non-constant classical values required to define the sequence must
    be provided as `pulse.sequence` arguments to allow a clean separation of
    classical, quantum and pulse value-passing processes within the MLIR.

    Pulse programs should be canonicalized such that all pulse
    operations occurr within a region contained by a `pulse.sequence` operation
    and are reached through a corresponding `qcs.remote_procedure`.

    Operations within the sequence cannot may capture values from outside its scope
    such as channel definitions. It is not isolated from above

    Syntax:

    ```
    op ::= `qcf.remote_prodedure` symbol-ref-id `(` argument-list `)` (`->`
    function-result-list)? function-attributes? region
    ```

    Example:

    ```mlir
    // External sequence function definitions.
    qcs.remote_procedure @x() {
        %0 = qcs.remote_procedure @measure_0(%arg2, %arg3, %arg4, %arg0) {pulse.acquireAt = 1056 : i64, pulse.timepoint = -8000 : i64, quir.noFastPathComm, quir.noJunoComm, quir.noJunoUse} : (!pulse.mixed_frame, !pulse.waveform, !pulse.mixed_frame, !pulse.mixed_frame) -> i1
        qcs.return
    }
    ```
  }];

  let arguments = (ins SymbolNameAttr:$sym_name,
                       TypeAttrOf<FunctionType>:$function_type,
                       OptionalAttr<StrAttr>:$sym_visibility,
                       OptionalAttr<DictArrayAttr>:$arg_attrs,
                       OptionalAttr<DictArrayAttr>:$res_attrs);

  //let results = (outs FunctionType);
  let regions = (region AnyRegion:$body);

  let builders = [OpBuilder<(ins
    "StringRef":$name, "FunctionType":$type,
    CArg<"ArrayRef<NamedAttribute>", "{}">:$attrs,
    CArg<"ArrayRef<DictionaryAttr>", "{}">:$argAttrs)
    >];
  let extraClassDeclaration = [{
    static RemoteProcedureOp create(Location location, StringRef name, FunctionType type,
                         ArrayRef<NamedAttribute> attrs = {});
    static RemoteProcedureOp create(Location location, StringRef name, FunctionType type,
                         Operation::dialect_attr_range attrs);
    static RemoteProcedureOp create(Location location, StringRef name, FunctionType type,
                         ArrayRef<NamedAttribute> attrs,
                         ArrayRef<DictionaryAttr> argAttrs);

    /// Create a deep copy of this sequence and all of its blocks, remapping any
    /// operands that use values outside of the sequence using the map that is
    /// provided (leaving them alone if no entry is present). If the mapper
    /// contains entries for sequence arguments, these arguments are not
    /// included in the new sequence. Replaces references to cloned sub-values
    /// with the corresponding value that is copied, and adds those mappings to
    /// the mapper.
    RemoteProcedureOp clone(IRMapping &mapper);
    RemoteProcedureOp clone();

    /// Clone the internal blocks and attributes from this sequence into dest.
    /// Any cloned blocks are appended to the back of dest. This sequence
    /// asserts that the attributes of the current sequence and dest are
    /// compatible.
    void cloneInto(RemoteProcedureOp dest, IRMapping &mapper);

    //===------------------------------------------------------------------===//
    // CallableOpInterface
    //===------------------------------------------------------------------===//

    /// Returns the region on the current operation that is callable. This may
    /// return null in the case of an external callable object, e.g. an external
    /// function.
    ::mlir::Region *getCallableRegion() { return isExternal() ? nullptr : &getBody(); }

    /// Returns the results types that the callable region produces when
    /// executed.
    ArrayRef<Type> getCallableResults() { return getFunctionType().getResults(); }

    /// Returns the argument attributes for all callable region arguments or
    /// null if there are none.
    ::mlir::ArrayAttr getCallableArgAttrs() {
      return getArgAttrs().value_or(nullptr);
    }

    /// Returns the result attributes for all callable region results or
    /// null if there are none.
    ::mlir::ArrayAttr getCallableResAttrs() {
      return getResAttrs().value_or(nullptr);
    }

    //===------------------------------------------------------------------===//
    // FunctionOpInterface Methods
    //===------------------------------------------------------------------===//

    /// Returns the argument types of this function.
    ArrayRef<Type> getArgumentTypes() { return getFunctionType().getInputs(); }

    /// Returns the result types of this function.
    ArrayRef<Type> getResultTypes() { return getFunctionType().getResults(); }

    //===------------------------------------------------------------------===//
    // SymbolOpInterface Methods
    //===------------------------------------------------------------------===//

    bool isDeclaration() { return isExternal(); }


  }];
  let hasCustomAssemblyFormat = 1;
  let hasVerifier = 1;

}



//===----------------------------------------------------------------------===//
// ReturnOp
//===----------------------------------------------------------------------===//

def QCS_ReturnOp : QCS_Op<"return", [Pure, Terminator, ReturnLike]>,
    Arguments<(ins Variadic<AnyClassical>:$operands)>, Results<(outs)> {
        let summary = "Terminator for qcs.remote_procedure.";
        let description = [{
                A terminator for regions that appear in the body of the `pulse.remote_procedure`
                operation. The operands to the `pulse.return` are the result values
                that are returned to the `qcs.remote_procedure` invocation of the target sequence.

                Example:

                ```mlir
                qcs.remote_procedure @foo(%angle: quir.angle<32>) -> (i1, i1)
                    ...
                    qcs.return %0, %1: i1, i1
                }
            }];
        let builders = [OpBuilder<(ins), [{ build($_builder, $_state, std::nullopt); }]>];
        let assemblyFormat = "attr-dict ($operands^ `:` type($operands))?";
        let hasVerifier = 1;
}



#endif // QCS_OPS
