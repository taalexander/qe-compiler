//===- TargetSystemRegistry.h - System Target Registry ----------*- C++ -*-===//
//
// (C) Copyright IBM 2023.
//
// Any modifications or derivative works of this code must retain this
// copyright notice, and modified files need to carry a notice indicating
// that they have been altered from the originals.
//
//===----------------------------------------------------------------------===//
//
//  Declaration of the QSSC target registry system.
//
//===----------------------------------------------------------------------===//
#ifndef TARGETSYSTEMREGISTRY_H
#define TARGETSYSTEMREGISTRY_H

#include "HAL/TargetSystem.h"

#include "Plugin/PluginRegistry.h"
#include "TargetSystemInfo.h"

namespace qssc::hal::registry::detail {
static llvm::Error noOptRegisterDialectFn(mlir::DialectRegistry &registry) {
  return llvm::Error::success();
}
static llvm::Error noOptRegisterFn() { return llvm::Error::success(); }
} // namespace qssc::hal::registry::detail

namespace qssc::hal::registry {

class TargetSystemRegistry
    : public qssc::plugin::registry::PluginRegistry<TargetSystemInfo> {
  using PluginRegistry =
      qssc::plugin::registry::PluginRegistry<TargetSystemInfo>;

public:
  TargetSystemRegistry(const TargetSystemRegistry &) = delete;
  void operator=(const TargetSystemRegistry &) = delete;

  /// @brief Register a specific target allocator with the QSSC system manually
  /// passing all arguments
  /// @tparam ConcreteTargetSystem system to register
  /// @param name The Name to register the target class under.
  /// @param description A description of the target system.
  /// @param targetFactory Function to construct the target.
  /// @param dialectRegistrar Function to registrar any dialects for the target.
  /// Will be called on compiler init.
  /// @param passRegistrar Function to register the target's MLIR passes.
  /// @param passPipelineRegistrar Function to register the target's pass
  /// pipelines.
  /// @return
  template <typename ConcreteTargetSystem>
  static bool registerPlugin(
      llvm::StringRef name, llvm::StringRef description,
      const TargetSystemInfo::PluginFactoryFunction &targetFactory,
      const TargetSystemInfo::DialectsFunction &dialectRegistrar =
          detail::noOptRegisterDialectFn,
      const TargetSystemInfo::PassesFunction &passRegistrar =
          detail::noOptRegisterFn,
      const TargetSystemInfo::PassPipelinesFunction &passPipelineRegistrar =
          detail::noOptRegisterFn) {
    return PluginRegistry::registerPlugin(name, name, description,
                                          targetFactory, dialectRegistrar,
                                          passRegistrar, passPipelineRegistrar);
  }

  /// @brief Register a TargetSystem with the system registry by implementing
  /// the required static methods (buildTarget, registerTargetPasses,
  /// registerTargetPipelines, registerTargetDialects) for the registered class.
  /// @tparam ConcreteTargetSystem the target system to register.
  /// @param name The Name to register the target class under.
  /// @param description A description of the target system.
  /// @return
  template <typename ConcreteTargetSystem>
  static bool registerPlugin(llvm::StringRef name,
                             llvm::StringRef description) {
    return PluginRegistry::registerPlugin(
        name, name, description, ConcreteTargetSystem::buildTarget,
        ConcreteTargetSystem::registerTargetDialects,
        ConcreteTargetSystem::registerTargetPasses,
        ConcreteTargetSystem::registerTargetPipelines);
  }

  static TargetSystemInfo *nullTargetSystemInfo();
};

} // namespace qssc::hal::registry

#endif
