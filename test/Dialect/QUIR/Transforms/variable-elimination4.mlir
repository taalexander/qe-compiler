// RUN: qss-compiler -X=mlir --quir-eliminate-variables %s --canonicalize| FileCheck %s --implicit-check-not '!quir.cbit' --implicit-check-not variable --implicit-check-not alloc --implicit-check-not store
//
// This test verifies store-forwarding for multi-bit registers.

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

// CHECK: module
module {
  oq3.declare_variable @success : i32
  oq3.declare_variable {output} @final : i32
  oq3.declare_variable @measure_result : !quir.cbit<1>
  oq3.declare_variable @is_excited : i32
  oq3.declare_variable {output} @result : i32

  func.func @main() -> i32 attributes {quir.classicalOnly = false} {
    %c17_i32 = arith.constant 17 : i32
    %c5_i32 = arith.constant 5 : i32
    %c1_i32 = arith.constant 1 : i32
    %c3_i32 = arith.constant 3 : i32
    %c20_i32 = arith.constant 20 : i32
    %false = arith.constant false
    %c0_i32 = arith.constant 0 : i32
    %c1 = arith.constant 1 : index
    %c1000 = arith.constant 1000 : index
    %c0 = arith.constant 0 : index
    qcs.init
    scf.for %arg0 = %c0 to %c1000 step %c1 {
      qcs.shot_init {qcs.num_shots = 1000 : i32}
      %0 = quir.declare_qubit {id = 0 : i32} : !quir.qubit<1>
      oq3.variable_assign @success : i32 = %c0_i32
      oq3.variable_assign @final : i32 = %c0_i32
      %1 = "oq3.cast"(%false) : (i1) -> !quir.cbit<1>
      oq3.variable_assign @measure_result : !quir.cbit<1> = %1
      oq3.variable_assign @is_excited : i32 = %c0_i32
      oq3.variable_assign @result : i32 = %c20_i32
      scf.for %arg1 = %c0_i32 to %c3_i32 step %c1_i32  : i32 {
        scf.while : () -> () {
          %6 = oq3.variable_load @success : i32
          // CHECK: blah
          %7 = arith.cmpi ne, %6, %c5_i32 : i32
          scf.condition(%7)
        } do {
          %6 = quir.measure(%0) {quir.noReportRuntime} : (!quir.qubit<1>) -> i1
          oq3.cbit_assign_bit @measure_result<1> [0] : i1 = %6
          %7 = oq3.variable_load @measure_result : !quir.cbit<1>
          %8 = "oq3.cast"(%7) : (!quir.cbit<1>) -> i32
          oq3.variable_assign @is_excited : i32 = %8
          %9 = oq3.variable_load @success : i32
          %10 = oq3.variable_load @is_excited : i32
          %11 = arith.addi %9, %10 : i32
          oq3.variable_assign @success : i32 = %11
          oq3.variable_assign @result : i32 = %c17_i32
          scf.yield
        } attributes {quir.classicalOnly = false}
        %3 = oq3.variable_load @final : i32
        %4 = oq3.variable_load @success : i32
        %5 = arith.addi %3, %4 : i32
        oq3.variable_assign @final : i32 = %5
        oq3.variable_assign @success : i32 = %c0_i32
      } {quir.classicalOnly = false, quir.physicalIds = [0 : i32]}
      %2 = quir.measure(%0) {quir.noReportUserResult} : (!quir.qubit<1>) -> i1
    } {qcs.shot_loop, quir.classicalOnly = false, quir.physicalIds = [0 : i32]}
    qcs.finalize
    return %c0_i32 : i32
  }
}
