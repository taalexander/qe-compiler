OPENQASM 3.0;
// RUN: qss-compiler -X=qasm --emit=ast-pretty %s | FileCheck %s --match-full-lines --check-prefix AST-PRETTY
// RUN: qss-compiler -X=qasm --emit=mlir %s --enable-circuits=false| FileCheck %s --match-full-lines --check-prefixes MLIR,MLIR-NO-CIRCUITS
// RUN: qss-compiler -X=qasm --emit=mlir %s --enable-circuits | FileCheck %s --match-full-lines --check-prefixes MLIR,MLIR-CIRCUITS

//
// This code is part of Qiskit.
//
// (C) Copyright IBM 2023.
//
// This code is licensed under the Apache License, Version 2.0 with LLVM
// Exceptions. You may obtain a copy of this license in the LICENSE.txt
// file in the root directory of this source tree.
//
// Any modifications or derivative works of this code must retain this
// copyright notice, and modified files need to carry a notice indicating
// that they have been altered from the originals.

int i = 15;
qubit $0;
// MLIR: quir.switch %{{.*}}{
// MLIR-NO-CIRCUITS: %angle = quir.constant #quir.angle<0.000000e+00> : !quir.angle<64>
// MLIR-NO-CIRCUITS: %angle_0 = quir.constant #quir.angle<1.000000e-01> : !quir.angle<64>
// MLIR-NO-CIRCUITS: %angle_1 = quir.constant #quir.angle<2.000000e-01> : !quir.angle<64>
// MLIR-NO-CIRCUITS: quir.builtin_U %{{.*}}, %angle, %angle_0, %angle_1 : !quir.qubit<1>, !quir.angle<64>, !quir.angle<64>, !quir.angle<64>
// MLIR-CIRCUITS:    quir.call_circuit @circuit_0(%0) : (!quir.qubit<1>) -> ()
// MLIR: }[1 : {
// MLIR: }2 : {
// MLIR: }3 : {
// MLIR: }5 : {
// MLIR: }12 : {
// MLIR: }]
// AST-PRETTY: SwitchStatementNode(SwitchQuantity(name=i, type=ASTTypeIdentifier),
switch (i) {
    // AST-PRETTY: statements=[
    // AST-PRETTY: CaseStatementNode(case=1, ),
    case 1: {
    }
    break;
    // AST-PRETTY: CaseStatementNode(case=2, ),
    case 2: {
    }
    break;
    // AST-PRETTY: CaseStatementNode(case=3, ),
    case 3: {
    }
    break;
    // AST-PRETTY: CaseStatementNode(case=5, ),
    case 5: {
    }
    break;
    // AST-PRETTY: CaseStatementNode(case=12, ),
    case 12: {
    }
    break;
    // AST-PRETTY: ],
    // AST-PRETTY: default statement=[
    // AST-PRETTY: UGateOpNode(params=[AngleNode(value=0.0, bits=64), AngleNode(value=0.1, bits=64), AngleNode(value=0.2, bits=64)], qubits=[], qcparams=[$0])
    // AST-PRETTY: ])
    default: {
        U(0.0, 0.1, 0.2) $0;
    }
    break;
}
