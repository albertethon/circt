//===- Statement.cpp - Slang statement conversion -------------------------===//
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

using namespace circt;
using namespace ImportVerilog;

LogicalResult Context::visitConditionalStmt(
    const slang::ast::ConditionalStatement *conditionalStmt) {
  auto loc = convertLocation(conditionalStmt->sourceRange.start());
  auto type = conditionalStmt->conditions.begin()->expr->type;

  Value cond = convertExpression(*conditionalStmt->conditions.begin()->expr);
  if (!cond)
    return failure();

  // The numeric value of the if expression is tested for being zero.
  // And if (expression) is equivalent to if (expression != 0).
  // So the following code is for handling `if (expression)`.
  if (!cond.getType().isa<mlir::IntegerType>()) {
    auto zeroValue =
        builder.create<moore::ConstantOp>(loc, convertType(*type), 0);
    cond = builder.create<moore::InEqualityOp>(loc, cond, zeroValue);
  }

  auto ifOp = builder.create<mlir::scf::IfOp>(
      loc, cond, conditionalStmt->ifFalse != nullptr);
  OpBuilder::InsertionGuard guard(builder);

  builder.setInsertionPoint(ifOp.thenYield());
  if (failed(convertStatement(&conditionalStmt->ifTrue)))
    return failure();

  if (conditionalStmt->ifFalse) {
    builder.setInsertionPoint(ifOp.elseYield());
    if (failed(convertStatement(conditionalStmt->ifFalse)))
      return failure();
  }

  return success();
}

// It can handle the statements like case, conditional(if), for loop, and etc.
LogicalResult
Context::convertStatement(const slang::ast::Statement *statement) {
  auto loc = convertLocation(statement->sourceRange.start());
  switch (statement->kind) {
  case slang::ast::StatementKind::Empty:
    return success();
  case slang::ast::StatementKind::List:
    for (auto *stmt : statement->as<slang::ast::StatementList>().list)
      if (failed(convertStatement(stmt)))
        return failure();
    break;
  case slang::ast::StatementKind::Block: {
    SymbolTableScopeT varScope(varSymbolTable);
    return convertStatement(&statement->as<slang::ast::BlockStatement>().body);
  }
  case slang::ast::StatementKind::ExpressionStatement:
    return success(convertExpression(
        statement->as<slang::ast::ExpressionStatement>().expr));
  case slang::ast::StatementKind::VariableDeclaration:
    return success();
  case slang::ast::StatementKind::Return:
    return mlir::emitError(loc, "unsupported statement: return");
  case slang::ast::StatementKind::Break:
    return mlir::emitError(loc, "unsupported statement: break");
  case slang::ast::StatementKind::Continue:
    return mlir::emitError(loc, "unsupported statement: continue");
  case slang::ast::StatementKind::Case:
    return mlir::emitError(loc, "unsupported statement: case");
  case slang::ast::StatementKind::PatternCase:
    return mlir::emitError(loc, "unsupported statement: pattern case");
  case slang::ast::StatementKind::ForLoop: {
    // reuse scf::whileOp to rewrite ForLoop
    // ------------
    // for (init_stmt; cond_expr; step_stmt) begin
    //   statements
    // end
    // -------------
    // init_stmt;
    // while (cond_expr) {
    //   body;
    //   step_stmt;
    // }
    // -------------

    const auto &forStmt = &statement->as<slang::ast::ForLoopStatement>();
    auto loc = convertLocation(forStmt->sourceRange.start());
    mlir::SmallVector<mlir::Type> types;
    auto type = convertType(*forStmt->stopExpr->type, loc);

    auto whileOp = builder.create<mlir::scf::WhileOp>(
        loc, types, mlir::SmallVector<Value, 0>{});
    OpBuilder::InsertionGuard guard(builder);

    // The before-region of the WhileOp.
    Block *before = builder.createBlock(&whileOp.getBefore());
    builder.setInsertionPointToEnd(before);
    Value cond = convertExpression(*forStmt->stopExpr);
    if (!cond)
      return failure();

    // The numeric value of the whileOp condition expression is tested for being
    // zero. And while (cond_expression) is equivalent to
    // while (cond_expression != 0). So the following code is for handling
    // `while (cond_expression)`.
    if (!cond.getType().isa<mlir::IntegerType>()) {
      auto zeroValue = builder.create<moore::ConstantOp>(loc, type, 0);
      cond = builder.create<moore::InEqualityOp>(loc, cond, zeroValue);
    }

    builder.create<mlir::scf::ConditionOp>(loc, cond, before->getArguments());

    // The after-region of the WhileOp.
    Block *after = builder.createBlock(&whileOp.getAfter());
    builder.setInsertionPointToStart(after);

    auto succeeded = convertStatement(&forStmt->body);
    //   step_stmt in forLoop
    for (auto *steps : forStmt->steps) {
      convertExpression(*steps);
    }
    builder.create<mlir::scf::YieldOp>(loc);
    return succeeded.success();
  }
  case slang::ast::StatementKind::RepeatLoop: {
    const auto &whileStmt = &statement->as<slang::ast::RepeatLoopStatement>();
    auto loc = convertLocation(whileStmt->sourceRange.start());
    auto type = convertType(*whileStmt->count.type, loc);
    Value countExpr = convertExpression(whileStmt->count);
    if (!countExpr)
      return failure();
    auto whileOp = builder.create<mlir::scf::WhileOp>(loc, type, countExpr);

    // The before-region of the WhileOp.
    Block *before = builder.createBlock(&whileOp.getBefore(), {}, type, loc);

    builder.setInsertionPointToEnd(before);
    Value cond;
    if (!countExpr.getType().isa<mlir::IntegerType>()) {
      auto zeroValue = builder.create<moore::ConstantOp>(loc, type, 0);
      cond = builder.create<moore::InEqualityOp>(loc, countExpr, zeroValue);
    }
    builder.create<mlir::scf::ConditionOp>(loc, cond, before->getArguments());

    // The after-region of the WhileOp.
    Block *after = builder.createBlock(&whileOp.getAfter(), {}, type, loc);
    builder.setInsertionPointToStart(after);

    auto succeeded = convertStatement(&whileStmt->body);

    // count decrement
    auto one = builder.create<moore::ConstantOp>(loc, type, 1);
    auto count = after->getArgument(0);
    auto result = builder.create<moore::SubOp>(loc, count, one);

    builder.create<mlir::scf::YieldOp>(loc, result->getResults());

    return succeeded.success();
  }
  case slang::ast::StatementKind::ForeachLoop:
    return mlir::emitError(loc, "unsupported statement: foreach loop");
  case slang::ast::StatementKind::WhileLoop: {
    const auto &whileStmt = &statement->as<slang::ast::WhileLoopStatement>();
    auto loc = convertLocation(whileStmt->sourceRange.start());
    mlir::SmallVector<mlir::Type> types;
    auto type = convertType(*whileStmt->cond.type, loc);

    auto whileOp = builder.create<mlir::scf::WhileOp>(
        loc, types, mlir::SmallVector<Value, 0>{});
    OpBuilder::InsertionGuard guard(builder);

    // The before-region of the WhileOp.
    Block *before = builder.createBlock(&whileOp.getBefore());
    builder.setInsertionPointToEnd(before);
    Value cond = convertExpression(whileStmt->cond);
    if (!cond)
      return failure();

    // The numeric value of the whileOp condition expression is tested for being
    // zero. And while (cond_expression) is equivalent to
    // while (cond_expression != 0). So the following code is for handling
    // `while (cond_expression)`.
    if (!cond.getType().isa<mlir::IntegerType>()) {
      auto zeroValue = builder.create<moore::ConstantOp>(loc, type, 0);
      cond = builder.create<moore::InEqualityOp>(loc, cond, zeroValue);
    }

    builder.create<mlir::scf::ConditionOp>(loc, cond, before->getArguments());

    // The after-region of the WhileOp.
    Block *after = builder.createBlock(&whileOp.getAfter());
    builder.setInsertionPointToStart(after);

    auto succeeded = convertStatement(&whileStmt->body);
    builder.create<mlir::scf::YieldOp>(loc);
    return succeeded.success();
  }
  case slang::ast::StatementKind::DoWhileLoop: {
    const auto &whileStmt = &statement->as<slang::ast::DoWhileLoopStatement>();
    auto loc = convertLocation(whileStmt->sourceRange.start());
    mlir::SmallVector<mlir::Type> types;
    auto type = convertType(*whileStmt->cond.type, loc);

    auto whileOp = builder.create<mlir::scf::WhileOp>(
        loc, types, mlir::SmallVector<Value, 0>{});
    OpBuilder::InsertionGuard guard(builder);

    // The before-region of the WhileOp.
    Block *before = builder.createBlock(&whileOp.getBefore());
    builder.setInsertionPointToEnd(before);

    auto succeeded = convertStatement(&whileStmt->body);
    Value cond = convertExpression(whileStmt->cond);
    if (!cond)
      return failure();
    // The numeric value of the whileOp condition expression is tested for being
    // zero. And while (cond_expression) is equivalent to
    // while (cond_expression != 0). So the following code is for handling
    // `while (cond_expression)`.
    if (!cond.getType().isa<mlir::IntegerType>()) {
      auto zeroValue = builder.create<moore::ConstantOp>(loc, type, 0);
      cond = builder.create<moore::InEqualityOp>(loc, cond, zeroValue);
    }

    builder.create<mlir::scf::ConditionOp>(loc, cond, before->getArguments());

    // The after-region of the WhileOp.
    Block *after = builder.createBlock(&whileOp.getAfter());
    builder.setInsertionPointToStart(after);

    builder.create<mlir::scf::YieldOp>(loc);
    return succeeded.success();
  }
  case slang::ast::StatementKind::ForeverLoop:
    return mlir::emitError(loc, "unsupported statement: forever loop");
  case slang::ast::StatementKind::Timed:
    if (failed(visitTimingControl(
            &statement->as<slang::ast::TimedStatement>().timing)))
      return failure();
    if (failed(convertStatement(
            &statement->as<slang::ast::TimedStatement>().stmt)))
      return failure();
    break;
  case slang::ast::StatementKind::ImmediateAssertion:
    return mlir::emitError(loc, "unsupported statement: immediate assertion");
  case slang::ast::StatementKind::ConcurrentAssertion:
    return mlir::emitError(loc, "unsupported statement: concurrent assertion");
  case slang::ast::StatementKind::DisableFork:
    return mlir::emitError(loc, "unsupported statement: disable fork");
  case slang::ast::StatementKind::Wait:
    return mlir::emitError(loc, "unsupported statement: wait");
  case slang::ast::StatementKind::WaitFork:
    return mlir::emitError(loc, "unsupported statement: wait fork");
  case slang::ast::StatementKind::WaitOrder:
    return mlir::emitError(loc, "unsupported statement: wait order");
  case slang::ast::StatementKind::EventTrigger:
    return mlir::emitError(loc, "unsupported statement: event trigger");
  case slang::ast::StatementKind::ProceduralAssign:
    return success(convertExpression(
        statement->as<slang::ast::ProceduralAssignStatement>().assignment));
  case slang::ast::StatementKind::ProceduralDeassign:
    return mlir::emitError(loc, "unsupported statement: procedural deassign");
  case slang::ast::StatementKind::RandCase:
    return mlir::emitError(loc, "unsupported statement: rand case");
  case slang::ast::StatementKind::RandSequence:
    return mlir::emitError(loc, "unsupported statement: rand sequence");
  case slang::ast::StatementKind::Conditional:
    return visitConditionalStmt(
        &statement->as<slang::ast::ConditionalStatement>());
  default:
    mlir::emitRemark(loc, "unsupported statement: ")
        << slang::ast::toString(statement->kind);
    return failure();
  }

  return success();
}
