//===- ClassicalOnlyDetection.cpp - detect pulse ops ------------*- C++ -*-===//
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
/// Defines pass for updating quir.classicalOnly flag based on the presence of
/// Pulse dialect Ops
///
//===----------------------------------------------------------------------===//

#include "Dialect/Pulse/Transforms/ClassicalOnlyDetection.h"

#include "Dialect/Pulse/IR/PulseDialect.h"
#include "Dialect/Pulse/IR/PulseOps.h"
#include "Dialect/QUIR/IR/QUIROps.h"

#include "mlir/Dialect/ControlFlow/IR/ControlFlowOps.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/Dominance.h"
#include "mlir/IR/Operation.h"
#include "mlir/IR/Visitors.h"
#include "mlir/Support/LLVM.h"

#include "llvm/ADT/StringRef.h"
#include "llvm/Support/Casting.h"

using namespace mlir;
using namespace mlir::pulse;

namespace mlir::pulse {

static const auto attrName = llvm::StringRef("quir.classicalOnly");

static void
markDominatorsOfPulseOp(OpBuilder &builder, mlir::DominanceInfo &domInfo,
                        std::vector<Operation *> &toAnalyzeIfDominates,
                        Operation *pulseOp) {
  // Iterate through attributing any operations that dominate the pulse op and
  // removing from the analysis list.
  std::vector<Operation *>::iterator iter;
  for (iter = toAnalyzeIfDominates.begin(); iter != toAnalyzeIfDominates.end();)
    if (domInfo.dominates(*iter, pulseOp)) {
      (*iter)->setAttr(attrName, builder.getBoolAttr(false));
      iter = toAnalyzeIfDominates.erase(iter);
    } else
      ++iter;
}

static bool hasPulseSubOps(Operation *inOp) {
  bool retVal = false;
  inOp->walk([&](Operation *op) {
    if (llvm::isa<pulse::PulseDialect>(op->getDialect())) {
      retVal = true;
      return WalkResult::interrupt();
    }
    return WalkResult::advance();
  });
  return retVal;
}

// Entry point for the ClassicalOnlyDetectionPass pass
void ClassicalOnlyDetectionPass::runOnOperation() {
  // This pass is only called on the top-level module Op

  auto &domInfo = getAnalysis<mlir::DominanceInfo>();

  Operation *moduleOperation = getOperation();
  OpBuilder builder(moduleOperation);

  std::vector<Operation *> toAnalyzeIfDominates;

  // Add operations to be verified for dominating a pulse operation
  moduleOperation->walk([&](Operation *op) {
    if (dyn_cast<scf::IfOp>(op) || dyn_cast<scf::ForOp>(op) ||
        dyn_cast<quir::SwitchOp>(op) || dyn_cast<SequenceOp>(op) ||
        dyn_cast<mlir::func::FuncOp>(op)) {
      // check for a pre-existing classicalOnly attribute
      // only update if the attribute does not exist or it is true
      // indicating that no quantum ops have been identified yet
      auto classicalOnlyAttr = op->getAttrOfType<BoolAttr>(attrName);
      if (!classicalOnlyAttr || classicalOnlyAttr.getValue())
        op->setAttr(attrName, builder.getBoolAttr(!hasPulseSubOps(op)));
      return WalkResult::advance();
    } else if (dyn_cast<cf::CondBranchOp>(op) || dyn_cast<cf::BranchOp>(op)) {
      auto classicalOnlyAttr = op->getAttrOfType<BoolAttr>(attrName);
      if (!classicalOnlyAttr || classicalOnlyAttr.getValue())
        // Classical only until proven otherwise
        op->setAttr(attrName, builder.getBoolAttr(true));
      toAnalyzeIfDominates.push_back(op);

      return WalkResult::advance();
    }

    // Attribute any operators that this pulse operation dominates
    if (llvm::isa<pulse::PulseDialect>(op->getDialect()) &&
        toAnalyzeIfDominates.size()) {
      markDominatorsOfPulseOp(builder, domInfo, toAnalyzeIfDominates, op);
    }
  });
} // ClassicalOnlyDetectionPass::runOnOperation

llvm::StringRef ClassicalOnlyDetectionPass::getArgument() const {
  return "pulse-classical-only-detection";
}
llvm::StringRef ClassicalOnlyDetectionPass::getDescription() const {
  return "Detect control flow blocks that contain only classical (non-quantum) "
         "operations, and decorate them with a classicalOnly bool attribute";
}

llvm::StringRef ClassicalOnlyDetectionPass::getName() const {
  return "Classical Only Detection Pass";
}
} // namespace mlir::pulse
