//===- DialectRegistration.h - Top-level dialect registration ---*- C++ -*-===//
//
// (C) Copyright IBM 2024.
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
//
//  This file declares the top-level functions for registering dialects
//
//===----------------------------------------------------------------------===//
#ifndef DIALECTREGISTRATION_H
#define DIALECTREGISTRATION_H

#include "mlir/IR/DialectRegistry.h"

#include "llvm/Support/Error.h"

namespace qssc::hal {
/// Register all registered MLIR target dialects
/// for the registered Targets with the
/// QSSC system.
llvm::Error registerTargetDialects(mlir::DialectRegistry &registry);

} // namespace qssc::hal

#endif // DIALECTREGISTRATION_H
