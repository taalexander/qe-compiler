//===- QCSOps.cpp - Quantum Control System dialect ops ----------*- C++ -*-===//
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
/// This file defines the operations in the Quantum Control System dialect.
///
//===----------------------------------------------------------------------===//

#include "Dialect/QCS/IR/QCSOps.h"

#include "Dialect/QCS/IR/QCSTypes.h"
#include "Dialect/QUIR/IR/QUIRAttributes.h"

#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/FunctionImplementation.h"
#include "mlir/IR/IRMapping.h"
#include "mlir/IR/Matchers.h"
#include "mlir/IR/OpImplementation.h"
#include "mlir/IR/OperationSupport.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/IR/SymbolTable.h"
#include "mlir/Support/LogicalResult.h"

#include "llvm/ADT/StringMap.h"

#include <cassert>

using namespace mlir;
using namespace mlir::qcs;

#define GET_OP_CLASSES
// NOLINTNEXTLINE(misc-include-cleaner): Required for MLIR registrations
#include "Dialect/QCS/IR/QCSOps.cpp.inc"

namespace {
LogicalResult
verifyQCSParameterOpSymbolUses(SymbolTableCollection &symbolTable,
                               mlir::Operation *op,
                               bool operandMustMatchSymbolType = false) {
  assert(op);

  // Check that op has attribute variable_name
  auto paramRefAttr = op->getAttrOfType<FlatSymbolRefAttr>("parameter_name");
  if (!paramRefAttr)
    return op->emitOpError(
        "requires a symbol reference attribute 'parameter_name'");

  // Check that symbol reference resolves to a parameter declaration
  auto declOp =
      symbolTable.lookupNearestSymbolFrom<DeclareParameterOp>(op, paramRefAttr);

  // check higher level modules
  if (!declOp) {
    auto targetModuleOp = op->getParentOfType<mlir::ModuleOp>();
    if (targetModuleOp) {
      auto topLevelModuleOp = targetModuleOp->getParentOfType<mlir::ModuleOp>();
      if (!declOp && topLevelModuleOp)
        declOp = symbolTable.lookupNearestSymbolFrom<DeclareParameterOp>(
            topLevelModuleOp, paramRefAttr);
    }
  }

  if (!declOp)
    return op->emitOpError() << "no valid reference to a parameter '"
                             << paramRefAttr.getValue() << "'";

  assert(op->getNumResults() <= 1 && "assume none or single result");

  // Check that type of variables matches result type of this Op
  if (op->getNumResults() == 1) {
    if (op->getResult(0).getType() != declOp.getType())
      return op->emitOpError(
          "type mismatch between variable declaration and variable use");
  }

  if (op->getNumOperands() > 0 && operandMustMatchSymbolType) {
    assert(op->getNumOperands() == 1 &&
           "type check only supported for a single operand");
    if (op->getOperand(0).getType() != declOp.getType())
      return op->emitOpError(
          "type mismatch between variable declaration and variable assignment");
  }
  return success();
}

} // anonymous namespace

//===----------------------------------------------------------------------===//
// ParameterLoadOp
//===----------------------------------------------------------------------===//

LogicalResult
ParameterLoadOp::verifySymbolUses(SymbolTableCollection &symbolTable) {
  return verifyQCSParameterOpSymbolUses(symbolTable, getOperation(), true);
}

// Returns the float value from the initial value of this parameter
ParameterType ParameterLoadOp::getInitialValue() {
  auto *op = getOperation();
  auto paramRefAttr =
      op->getAttrOfType<mlir::FlatSymbolRefAttr>("parameter_name");
  auto declOp =
      mlir::SymbolTable::lookupNearestSymbolFrom<mlir::qcs::DeclareParameterOp>(
          op, paramRefAttr);

  // check higher level modules

  auto currentScopeOp = op->getParentOfType<mlir::ModuleOp>();
  do {
    declOp = mlir::SymbolTable::lookupNearestSymbolFrom<
        mlir::qcs::DeclareParameterOp>(currentScopeOp, paramRefAttr);
    if (declOp)
      break;
    currentScopeOp = currentScopeOp->getParentOfType<mlir::ModuleOp>();
    assert(currentScopeOp);
  } while (!declOp);

  assert(declOp);

  double retVal;

  auto iniValue = declOp.getInitialValue();
  if (iniValue.has_value()) {
    auto angleAttr = iniValue.value().dyn_cast<mlir::quir::AngleAttr>();

    auto floatAttr = iniValue.value().dyn_cast<FloatAttr>();

    if (!(angleAttr || floatAttr)) {
      op->emitError(
          "Parameters are currently limited to angles or float[64] only.");
      return 0.0;
    }

    if (angleAttr)
      retVal = angleAttr.getValue().convertToDouble();

    if (floatAttr)
      retVal = floatAttr.getValue().convertToDouble();

    return retVal;
  }

  op->emitError("Does not have initial value set.");
  return 0.0;
}

// Returns the float value from the initial value of this parameter
// this version uses a precomputed map of parameter_name to the initial_value
// in order to avoid slow SymbolTable lookups
ParameterType ParameterLoadOp::getInitialValue(
    llvm::StringMap<ParameterType> &declareParametersMap) {
  auto *op = getOperation();
  auto paramRefAttr =
      op->getAttrOfType<mlir::FlatSymbolRefAttr>("parameter_name");

  auto paramOpEntry = declareParametersMap.find(paramRefAttr.getValue());

  if (paramOpEntry == declareParametersMap.end()) {
    op->emitError("Could not find declare parameter op " +
                  paramRefAttr.getValue().str());
    return 0.0;
  }

  return paramOpEntry->second;
}

//===----------------------------------------------------------------------===//
// ParallelEndOp
//===----------------------------------------------------------------------===//

mlir::ParseResult ParallelEndOp::parse(mlir::OpAsmParser &parser,
                                       mlir::OperationState &result) {
  return mlir::success();
}

void ParallelEndOp::print(mlir::OpAsmPrinter &printer) {
  printer << getOperationName();
}

//===----------------------------------------------------------------------===//
// LaunchDistributedProcedureOp
//===----------------------------------------------------------------------===//

auto LaunchDistributedProcedureOp ::getCalleeType() -> FunctionType {
  return FunctionType::get(getContext(), getOperandTypes(), getResultTypes());
}

LogicalResult LaunchDistributedProcedureOp ::verifySymbolUses(
    SymbolTableCollection &symbolTable) {

  auto procedureAttr = (*this)->getAttrOfType<FlatSymbolRefAttr>("callee");
  if (!procedureAttr)
    return emitOpError("Requires a 'callee' symbol reference attribute");

  auto procedure = symbolTable.lookupNearestSymbolFrom<DistributedProcedureOp>(
      *this, procedureAttr);
  if (!procedure)
    return emitOpError() << "'" << procedureAttr.getValue()
                         << "' does not reference a valid procedure";

  // Verify the types match
  auto procedureType = procedure.getFunctionType();

  if (procedureType.getNumInputs() != getNumOperands())
    return emitOpError("incorrect number of operands for the callee procedure");

  for (unsigned i = 0; i != procedureType.getNumInputs(); ++i) {
    if (getOperand(i).getType() != procedureType.getInput(i)) {
      auto diag = emitOpError("operand type mismatch at index ") << i;
      diag.attachNote() << "op input types: " << getOperandTypes();
      diag.attachNote() << "function operand types: "
                        << procedureType.getInputs();
      return diag;
    }
  }

  if (procedureType.getNumResults() != getNumResults())
    return emitOpError("incorrect number of results for the callee procedure");

  for (unsigned i = 0; i != procedureType.getNumResults(); ++i) {
    if (getResult(i).getType() != procedureType.getResult(i)) {
      auto diag = emitOpError("result type mismatch at index ") << i;
      diag.attachNote() << "op result types: " << getResultTypes();
      diag.attachNote() << "function result types: "
                        << procedureType.getResults();
      return diag;
    }
  }

  return success();
}

//===----------------------------------------------------------------------===//
// LaunchDistributedOp
//===----------------------------------------------------------------------===//

void LaunchDistributedOp::build(OpBuilder &builder, OperationState &result) {
  // Create a procedure body region
  Region *kernelRegion = result.addRegion();
  Block *body = new Block();
  kernelRegion->push_back(body);
}

LogicalResult LaunchDistributedOp::verifyRegions() {

  // Block terminators without successors are expected to exit the kernel region
  // and must be `qcs.terminator`.
  for (Block &block : getBody()) {
    if (block.empty())
      continue;
    if (block.back().getNumSuccessors() != 0)
      continue;
    if (!isa<qcs::TerminatorOp>(&block.back())) {
      return block.back()
          .emitError()
          .append("expected '", qcs::TerminatorOp::getOperationName(),
                  "' or a terminator with successors")
          .attachNote(getLoc())
          .append("in '", LaunchDistributedOp::getOperationName(),
                  "' body region");
    }
  }

  return success();
}

//===----------------------------------------------------------------------===//
// DistributedProcedureOp
//
// This code section was derived and modified from the LLVM project FuncOp
// Consequently it is licensed as Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

mlir::ParseResult DistributedProcedureOp::parse(mlir::OpAsmParser &parser,
                                                mlir::OperationState &result) {
  auto buildFuncType =
      [](Builder &builder, ArrayRef<Type> argTypes, ArrayRef<Type> results,
         function_interface_impl::VariadicFlag,
         std::string &) { return builder.getFunctionType(argTypes, results); };

  return function_interface_impl::parseFunctionOp(
      parser, result, /*allowVariadic=*/false,
      getFunctionTypeAttrName(result.name), buildFuncType,
      getArgAttrsAttrName(result.name), getResAttrsAttrName(result.name));
}

void DistributedProcedureOp::print(mlir::OpAsmPrinter &printer) {
  function_interface_impl::printFunctionOp(
      printer, *this, /*isVariadic=*/false, getFunctionTypeAttrName(),
      getArgAttrsAttrName(), getResAttrsAttrName());
}

namespace {
/// Verify the argument list and entry block are in agreement.
LogicalResult verifyArgumentAndEntry_(DistributedProcedureOp op) {
  auto fnInputTypes = op.getFunctionType().getInputs();
  Block &entryBlock = op.front();
  for (unsigned i = 0; i != entryBlock.getNumArguments(); ++i)
    if (fnInputTypes[i] != entryBlock.getArgument(i).getType())
      return op.emitOpError("type of entry block argument #")
             << i << '(' << entryBlock.getArgument(i).getType()
             << ") must match the type of the corresponding argument in "
             << "function signature(" << fnInputTypes[i] << ')';
  return success();
}
} // anonymous namespace

LogicalResult DistributedProcedureOp::verify() {
  // If external will be linked in later and nothing to do
  if (isExternal())
    return success();

  if (failed(verifyArgumentAndEntry_(*this)))
    return mlir::failure();

  return success();
}

DistributedProcedureOp
DistributedProcedureOp::create(Location location, StringRef name,
                               FunctionType type,
                               ArrayRef<NamedAttribute> attrs) {
  OpBuilder builder(location->getContext());
  OperationState state(location, getOperationName());
  DistributedProcedureOp::build(builder, state, name, type, attrs);
  return cast<DistributedProcedureOp>(Operation::create(state));
}
DistributedProcedureOp
DistributedProcedureOp::create(Location location, StringRef name,
                               FunctionType type,
                               Operation::dialect_attr_range attrs) {
  SmallVector<NamedAttribute, 8> const attrRef(attrs);
  return create(location, name, type, attrRef);
}
DistributedProcedureOp DistributedProcedureOp::create(
    Location location, StringRef name, FunctionType type,
    ArrayRef<NamedAttribute> attrs, ArrayRef<DictionaryAttr> argAttrs) {
  DistributedProcedureOp circ = create(location, name, type, attrs);
  circ.setAllArgAttrs(argAttrs);
  return circ;
}

void DistributedProcedureOp::build(OpBuilder &builder, OperationState &state,
                                   StringRef name, FunctionType type,
                                   ArrayRef<NamedAttribute> attrs,
                                   ArrayRef<DictionaryAttr> argAttrs) {
  state.addAttribute(SymbolTable::getSymbolAttrName(),
                     builder.getStringAttr(name));
  state.addAttribute(getFunctionTypeAttrName(state.name), TypeAttr::get(type));
  state.attributes.append(attrs.begin(), attrs.end());
  state.addRegion();

  if (argAttrs.empty())
    return;
  assert(type.getNumInputs() == argAttrs.size());
  function_interface_impl::addArgAndResultAttrs(
      builder, state, argAttrs, /*resultAttrs=*/std::nullopt,
      getArgAttrsAttrName(state.name), getResAttrsAttrName(state.name));
}

/// Clone the internal blocks and attributes from this procedure to the
/// destination procedure.
void DistributedProcedureOp::cloneInto(DistributedProcedureOp dest,
                                       IRMapping &mapper) {
  // Add the attributes of this function to dest.
  llvm::MapVector<StringAttr, Attribute> newAttrMap;
  for (const auto &attr : dest->getAttrs())
    newAttrMap.insert({attr.getName(), attr.getValue()});
  for (const auto &attr : (*this)->getAttrs())
    newAttrMap.insert({attr.getName(), attr.getValue()});

  auto newAttrs = llvm::to_vector(llvm::map_range(
      newAttrMap, [](std::pair<StringAttr, Attribute> attrPair) {
        return NamedAttribute(attrPair.first, attrPair.second);
      }));
  dest->setAttrs(DictionaryAttr::get(getContext(), newAttrs));

  // Clone the body.
  getBody().cloneInto(&dest.getBody(), mapper);
}

/// Create a deep copy of this procedure and all of its block.
/// Remap any operands that use values outside of the function
/// Using the provider mapper. Replace references to
/// cloned sub-values with the corresponding copied value and
/// add to the mapper
DistributedProcedureOp DistributedProcedureOp::clone(IRMapping &mapper) {
  FunctionType newType = getFunctionType();

  // If the function contains a body, then its possible arguments
  // may be deleted in the mapper. Verify this so they aren't
  // added to the input type vector.
  bool const isExternalSequence = isExternal();
  if (!isExternalSequence) {
    SmallVector<Type, 4> inputTypes;
    inputTypes.reserve(newType.getNumInputs());
    for (unsigned i = 0; i != getNumArguments(); ++i)
      if (!mapper.contains(getArgument(i)))
        inputTypes.push_back(newType.getInput(i));
    newType = FunctionType::get(getContext(), inputTypes, newType.getResults());
  }

  // Create the new procedure
  DistributedProcedureOp newSeq =
      cast<DistributedProcedureOp>(getOperation()->cloneWithoutRegions());
  newSeq.setType(newType);

  // Clone the current function into the new one and return.
  cloneInto(newSeq, mapper);
  return newSeq;
}

DistributedProcedureOp DistributedProcedureOp::clone() {
  IRMapping mapper;
  return clone(mapper);
}

//===----------------------------------------------------------------------===//
// ReturnOp
//
// This code section was derived and modified from the LLVM project's standard
// dialect ReturnOp. Consequently it is licensed as described below.
//
//===----------------------------------------------------------------------===//

LogicalResult ReturnOp::verify() {
  auto procedure = (*this)->getParentOfType<DistributedProcedureOp>();
  FunctionType const procedureType = procedure.getFunctionType();

  auto numResults = procedureType.getNumResults();
  // Verify number of operands match type signature
  if (numResults != getOperands().size()) {
    return emitError()
        .append("expected ", numResults, " result operands")
        .attachNote(procedure.getLoc())
        .append("return type declared here");
  }

  int i = 0;
  for (const auto [type, operand] :
       llvm::zip(procedureType.getResults(), getOperands())) {
    auto opType = operand.getType();
    if (type != opType) {
      return emitOpError() << "unexpected type `" << opType << "' for operand #"
                           << i;
    }
    i++;
  }
  return success();
}
