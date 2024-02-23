OPENQASM 3.0;
// RUN: qss-compiler -X=qasm --emit=ast-pretty %s | FileCheck %s --check-prefix AST-PRETTY
// RUN: qss-compiler -X=qasm --emit=mlir %s | FileCheck %s --check-prefix MLIR



int i0 = 1;
int i1 = 2;
int i2;

float f0 = 1.0;
float f1 = 2.0;
float f2;


// Operator precedence

int i3 = 3;
int i4 = 4;

float f3 = 3.0;



i2 = i0 * i1 ** i3 * i4;
// WARNING: Operator precedence of power is incorrect.
// TODO: power should take precedence over multiply.
// MLIR: %[[i0:.*]] = oq3.variable_load @i0 : i32
// MLIR: %[[i1:.*]] = oq3.variable_load @i1 : i32
// MLIR: %[[mul:.*]] = arith.muli %[[i0]], %[[i1]] : i32
// MLIR: %[[i3:.*]] = oq3.variable_load @i3 : i32
// MLIR: %[[pow:.*]] = math.ipowi %[[mul]], %[[i3]] : i32
// MLIR: oq3.variable_assign @i2 : i32 = %[[pow]]



