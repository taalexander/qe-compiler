//===- VariableElimination.h - Lower and eliminate variables ----*- C++ -*-===//
//
// (C) Copyright IBM 2022.
//
// Any modifications or derivative works of this code must retain this
// copyright notice, and modified files need to carry a notice indicating
// that they have been altered from the originals.
//
//===----------------------------------------------------------------------===//
///
///  This file declares the passes for converting QUIR variables to memref
///  operations and eliminating them where possible.
///
//===----------------------------------------------------------------------===//

#ifndef QUIR_VARIABLE_ELIMINATION_H
#define QUIR_VARIABLE_ELIMINATION_H

#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Pass/Pass.h"

namespace mlir::quir {

struct VariableEliminationPass
    : public PassWrapper<VariableEliminationPass, OperationPass<>> {
  void runOnOperation() override;

  bool externalizeOutputVariables;

  VariableEliminationPass(bool externalizeOutputVariables = false)
      : PassWrapper(), externalizeOutputVariables(externalizeOutputVariables) {}

  llvm::StringRef getArgument() const override;
  llvm::StringRef getDescription() const override;

  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<mlir::memref::MemRefDialect>();
    registry.insert<mlir::AffineDialect>();
  }
}; // struct VariableEliminationPass

} // namespace mlir::quir

#endif // QUIR_VARIABLE_ELIMINATION_H