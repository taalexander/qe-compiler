OPENQASM 3.0;
// RUN: qss-compiler -X=qasm --emit=ast-pretty %s | FileCheck %s --match-full-lines --check-prefix AST-PRETTY
// RUN: qss-compiler -X=qasm --emit=mlir %s | FileCheck %s --match-full-lines --check-prefix MLIR

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

// AST-PRETTY: DeclarationNode(type=ASTTypeBitset, CBitNode(name=bs, bits=32, value=00000000000000000000000000000001))
bit[32] bs = 1;

// AST-PRETTY: ForStatementNode(start=0, end=4,
// MLIR: scf.for %arg0 = %c0 to %c1000 step %c1 {
for i in [0 : 4] {
    // AST-PRETTY: statements=
    // AST-PRETTY: BinaryOpNode(type=ASTOpTypeAssign, left=IdentifierNode(name=bs, bits=32), right=BinaryOpNode(type=ASTOpTypeBitOr, left=IdentifierNode(name=bs, bits=32), right=IdentifierNode(name=i, bits=32))
    // AST-PRETTY: )
    // MLIR: %1 = oq3.variable_load @bs : !quir.cbit<32>
    // MLIR: %2 = "oq3.cast"(%arg1) : (i32) -> !quir.cbit<32>
    // MLIR: %3 = oq3.cbit_or %1, %2 : !quir.cbit<32>
    // MLIR: oq3.variable_assign @bs : !quir.cbit<32> = %3
    bs = bs | i;
}

// AST-PRETTY: ForStatementNode(start=0, end=3,
// MLIR: scf.for %arg1 = %c0_i32_2 to %c4_i32 step %c1_i32_3  : i32 {
for i in [0 : 3] {
    // AST-PRETTY: statements=
    // AST-PRETTY: BinaryOpNode(type=ASTOpTypeAssign, left=IdentifierNode(name=bs, bits=32), right=BinaryOpNode(type=ASTOpTypeBitAnd, left=IdentifierNode(name=bs, bits=32), right=IdentifierNode(name=i, bits=32))
    // AST-PRETTY: )
    // MLIR: %1 = oq3.variable_load @bs : !quir.cbit<32>
    // MLIR: %2 = "oq3.cast"(%arg1) : (i32) -> !quir.cbit<32>
    // MLIR: %3 = oq3.cbit_and %1, %2 : !quir.cbit<32>
    // MLIR: oq3.variable_assign @bs : !quir.cbit<32> = %3
    bs = bs & i;
}

// AST-PRETTY: ForStatementNode(start=0, end=5,
// MLIR: scf.for %arg1 = %c0_i32_4 to %c6_i32 step %c1_i32_5  : i32 {
for i in [0 : 5] {
    // AST-PRETTY: statements=
    // AST-PRETTY: BinaryOpNode(type=ASTOpTypeAssign, left=IdentifierNode(name=bs, bits=32), right=BinaryOpNode(type=ASTOpTypeXor, left=IdentifierNode(name=bs, bits=32), right=IdentifierNode(name=i, bits=32))
    // AST-PRETTY: )
    // MLIR: %1 = oq3.variable_load @bs : !quir.cbit<32>
    // MLIR: %2 = "oq3.cast"(%arg1) : (i32) -> !quir.cbit<32>
    // MLIR: %3 = oq3.cbit_xor %1, %2 : !quir.cbit<32>
    // MLIR: oq3.variable_assign @bs : !quir.cbit<32> = %3
    bs = bs ^ i;
}
