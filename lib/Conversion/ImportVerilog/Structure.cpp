//===- Structure.cpp - Slang hierarchy conversion -------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "ImportVerilogInternals.h"
#include "slang/ast/ASTVisitor.h"
#include "slang/ast/Symbol.h"
#include "slang/ast/symbols/CompilationUnitSymbols.h"
#include "slang/ast/symbols/InstanceSymbols.h"
#include "slang/ast/symbols/VariableSymbols.h"
#include "slang/ast/types/AllTypes.h"
#include "slang/ast/types/Type.h"
#include "slang/syntax/SyntaxVisitor.h"
#include "llvm/ADT/StringRef.h"

using namespace circt;
using namespace ImportVerilog;

LogicalResult
Context::convertCompilation(slang::ast::Compilation &compilation) {
  auto &root = compilation.getRoot();

  // Visit all the compilation units. This will mainly cover non-instantiable
  // things like packages.
  for (auto *unit : root.compilationUnits)
    for (auto &member : unit->members())
      LLVM_DEBUG(llvm::dbgs() << "Converting symbol " << member.name << "\n");

  // Prime the root definition worklist by adding all the top-level modules to
  // it.
  for (auto *inst : root.topInstances)
    convertModuleHeader(&inst->body);

  // Convert all the root module definitions.
  while (!moduleWorklist.empty()) {
    auto *module = moduleWorklist.front();
    moduleWorklist.pop();
    if (failed(convertModuleBody(module)))
      return failure();
  }

  return success();
}

Operation *
Context::convertModuleHeader(const slang::ast::InstanceBodySymbol *module) {
  if (auto *op = moduleOps.lookup(module))
    return op;
  auto loc = convertLocation(module->location);

  // We only support modules for now. Extension to interfaces and programs
  // should be trivial though, since they are essentially the same thing with
  // only minor differences in semantics.
  if (module->getDefinition().definitionKind !=
      slang::ast::DefinitionKind::Module) {
    mlir::emitError(loc, "unsupported construct: ")
        << module->getDefinition().getKindString();
    return nullptr;
  }

  // Handle the port list.
  LLVM_DEBUG(llvm::dbgs() << "Ports of module " << module->name << "\n");
  for (auto *symbol : module->getPortList()) {
    auto portLoc = convertLocation(symbol->location);
    auto *port = symbol->as_if<slang::ast::PortSymbol>();
    if (!port) {
      mlir::emitError(portLoc, "unsupported port: `")
          << symbol->name << "` (" << slang::ast::toString(symbol->kind) << ")";
      return nullptr;
    }
    LLVM_DEBUG(llvm::dbgs() << "- " << port->name << " "
                            << slang::ast::toString(port->direction) << "\n");
    if (auto *intSym = port->internalSymbol) {
      LLVM_DEBUG(llvm::dbgs() << "  - Internal symbol " << intSym->name << " ("
                              << slang::ast::toString(intSym->kind) << ")\n");
    }
    if (auto *expr = port->getInternalExpr()) {
      LLVM_DEBUG(llvm::dbgs() << "  - Internal expr "
                              << slang::ast::toString(expr->kind) << "\n");
    }
  }

  // Create an empty module that corresponds to this module.
  auto moduleOp = rootBuilder.create<moore::SVModuleOp>(loc, module->name);
  moduleOp.getBody().emplaceBlock();

  // Add the module to the symbol table of the MLIR module, which uniquifies its
  // name as we'd expect.
  symbolTable.insert(moduleOp);

  // Schedule the body to be lowered.
  moduleWorklist.push(module);
  moduleOps.insert({module, moduleOp});
  return moduleOp;
}

LogicalResult
Context::convertModuleBody(const slang::ast::InstanceBodySymbol *module) {
  LLVM_DEBUG(llvm::dbgs() << "Converting body of module " << module->name
                          << "\n");
  auto *moduleOp = moduleOps.lookup(module);
  assert(moduleOp);
  auto builder =
      OpBuilder::atBlockEnd(&cast<moore::SVModuleOp>(moduleOp).getBodyBlock());

  // Create a new scope in a module. When the processing of a module is
  // terminated, the scope is destroyed and the mappings created in this scope
  // are dropped.
  SymbolTableScopeT varScope(varSymbolTable);

  for (auto &member : module->members()) {
    LLVM_DEBUG(llvm::dbgs()
               << "- Handling " << slang::ast::toString(member.kind) << "\n");
    auto loc = convertLocation(member.location);

    // Skip parameters. The AST is already monomorphized.
    if (member.kind == slang::ast::SymbolKind::Parameter)
      continue;

    // Skip type-related declarations. These are absorbedby the types.
    if (member.kind == slang::ast::SymbolKind::TypeAlias ||
        member.kind == slang::ast::SymbolKind::TypeParameter ||
        member.kind == slang::ast::SymbolKind::TransparentMember)
      continue;

    // Handle instances.
    if (auto *instAst = member.as_if<slang::ast::InstanceSymbol>()) {
      auto *targetModule = convertModuleHeader(&instAst->body);
      if (!targetModule)
        return failure();
      builder.create<moore::InstanceOp>(
          loc, builder.getStringAttr(instAst->name),
          FlatSymbolRefAttr::get(SymbolTable::getSymbolName(targetModule)));
      continue;
    }

    // Handle variables.
    if (auto *varAst = member.as_if<slang::ast::VariableSymbol>()) {
      auto loweredType = convertType(*varAst->getDeclaredType());
      if (!loweredType)
        return failure();
      rootBuilder.setInsertionPointToEnd(builder.getInsertionBlock());
      Value varOp = rootBuilder.create<moore::VariableOp>(
          convertLocation(varAst->location), loweredType,
          builder.getStringAttr(varAst->name));
      if (varAst->getInitializer()) {
        Value value = visitExpression(varAst->getInitializer());
        if (value.getType() != loweredType)
          value =
              rootBuilder.create<moore::ConversionOp>(loc, loweredType, value);
        rootBuilder.create<moore::BPAssignOp>(loc, varOp, value);
      }
      varSymbolTable.insert(varAst->name, varOp);
      continue;
    }

    // Handle Nets.
    if (auto *netAst = member.as_if<slang::ast::NetSymbol>()) {
      auto loweredType = convertType(*netAst->getDeclaredType());
      if (!loweredType)
        return failure();
      rootBuilder.setInsertionPointToEnd(builder.getInsertionBlock());
      Value netOp = rootBuilder.create<moore::NetOp>(
          convertLocation(netAst->location), loweredType,
          builder.getStringAttr(netAst->name),
          builder.getStringAttr(netAst->netType.name));
      if (netAst->getInitializer()) {
        Value value = visitExpression(netAst->getInitializer());
        if (value.getType() != loweredType)
          value =
              rootBuilder.create<moore::ConversionOp>(loc, loweredType, value);
        rootBuilder.create<moore::CAssignOp>(loc, netOp, value);
      }
      varSymbolTable.insert(netAst->name, netOp);
      continue;
    }

    // Handle Ports.
    if (auto *portAst = member.as_if<slang::ast::PortSymbol>()) {
      auto loweredType = convertType(portAst->getType());
      if (!loweredType)
        return failure();
      builder.create<moore::PortOp>(
          convertLocation(portAst->location),
          builder.getStringAttr(portAst->name),
          static_cast<moore::Direction>(portAst->direction));
      continue;
    }

    // Handle AssignOp.
    if (auto *assignAst = member.as_if<slang::ast::ContinuousAssignSymbol>()) {
      rootBuilder.setInsertionPointToEnd(builder.getBlock());
      visitAssignmentExpr(
          &assignAst->getAssignment().as<slang::ast::AssignmentExpression>());
      continue;
    }

    // Handle ProceduralBlock.
    if (auto *procAst = member.as_if<slang::ast::ProceduralBlockSymbol>()) {
      auto loc = convertLocation(procAst->location);
      rootBuilder.setInsertionPointToEnd(
          &builder
               .create<moore::ProcedureOp>(
                   loc, static_cast<moore::Procedure>(procAst->procedureKind))
               .getBodyBlock());
      convertStatement(&procAst->getBody());
      continue;
    }

    // Otherwise just report that we don't support this SV construct yet and
    // skip over it. We'll want to make this an error, but in the early phases
    // we'll just want to cover ground as quickly as possible and skip over
    // things we don't support.
    mlir::emitWarning(loc, "unsupported construct ignored: ")
        << slang::ast::toString(member.kind);
  }

  return success();
}
