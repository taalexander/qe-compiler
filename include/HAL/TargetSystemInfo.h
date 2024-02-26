//===- TargetSystemInfo.h - System Target Registry --------------*- C++ -*-===//
//
// (C) Copyright IBM 2023.
//
// Any modifications or derivative works of this code must retain this
// copyright notice, and modified files need to carry a notice indicating
// that they have been altered from the originals.
//
//===----------------------------------------------------------------------===//
//
//  Declaration of the QSSC target system info.
//
//===----------------------------------------------------------------------===//
#ifndef TARGETSYSTEMINFO_H
#define TARGETSYSTEMINFO_H

#include "Arguments/Arguments.h"
#include "HAL/TargetSystem.h"
#include "Plugin/PluginInfo.h"
#include "Support/Pimpl.h"

#include "mlir/IR/DialectRegistry.h"

namespace qssc::hal::registry {

/// Class to group info about a registered target. Such as how to invoke
/// and a description.
class TargetSystemInfo
    : public qssc::plugin::registry::PluginInfo<qssc::hal::TargetSystem> {

public:
  using PluginInfo =
      qssc::plugin::registry::PluginInfo<qssc::hal::TargetSystem>;
  using DialectsFunction =
      std::function<llvm::Error(mlir::DialectRegistry &registry)>;
  using PassesFunction = std::function<llvm::Error()>;
  using PassPipelinesFunction = std::function<llvm::Error()>;

  /// Construct this entry
  TargetSystemInfo(llvm::StringRef name, llvm::StringRef description,
                   PluginInfo::PluginFactoryFunction targetFactory,
                   DialectsFunction dialectRegistrar,
                   PassesFunction passRegistrar,
                   PassPipelinesFunction passPipelineRegistrar);

  ~TargetSystemInfo();

  /// Create the target system and register it under the given context.
  llvm::Expected<qssc::hal::TargetSystem *>
  createTarget(mlir::MLIRContext *context,
               std::optional<PluginInfo::PluginConfiguration> configuration);

  /// Get the target system registered for the given context. First checks for
  /// a target registered exactly for the given context. If no such context is
  /// found, checks if a target is registered under nullptr, and returns
  /// that. If no target is found, an error is returned.
  llvm::Expected<qssc::hal::TargetSystem *>
  getTarget(mlir::MLIRContext *context) const;

  /// Register this target's MLIR dialects with the QSSC system.
  /// Should only be called once on initialization.
  llvm::Error registerTargetDialects(mlir::DialectRegistry &registry) const;

  /// Register this target's MLIR passes with the QSSC system.
  /// Should only be called once on initialization.
  llvm::Error registerTargetPasses() const;

  /// Register this target's MLIR passe pipelines with the QSSC system.
  /// Should only be called once on initialization.
  llvm::Error registerTargetPassPipelines() const;

private:
  struct Impl;

  qssc::support::Pimpl<Impl> impl;

  DialectsFunction dialectRegistrar;

  PassesFunction passRegistrar;

  PassPipelinesFunction passPipelineRegistrar;
};

} // namespace qssc::hal::registry

#endif
