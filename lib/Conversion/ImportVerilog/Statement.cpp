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

namespace {
struct StmtVisitor {
  Context &context;
  Location loc;
  OpBuilder &builder;
  LogicalResult visit(const slang::ast::ConditionalStatement &conditionalStmt) {
    auto type = conditionalStmt.conditions.begin()->expr->type;

    Value cond =
        context.convertExpression(*conditionalStmt.conditions.begin()->expr);
    if (!cond)
      return failure();

    // The numeric value of the if expression is tested for being zero.
    // And if (expression) is equivalent to if (expression != 0).
    // So the following code is for handling `if (expression)`.
    if (!cond.getType().isa<mlir::IntegerType>()) {
      auto zeroValue =
          builder.create<moore::ConstantOp>(loc, context.convertType(*type), 0);
      cond = builder.create<moore::InEqualityOp>(loc, cond, zeroValue);
    }

    auto ifOp = builder.create<mlir::scf::IfOp>(
        loc, cond, conditionalStmt.ifFalse != nullptr);
    OpBuilder::InsertionGuard guard(builder);

    builder.setInsertionPoint(ifOp.thenYield());
    if (conditionalStmt.ifTrue.visit(*this).failed())
      return failure();

    if (conditionalStmt.ifFalse) {
      builder.setInsertionPoint(ifOp.elseYield());
      if (conditionalStmt.ifFalse->visit(*this).failed())
        return failure();
    }

    return success();
  }

  LogicalResult
  visit(const slang::ast::ProceduralAssignStatement &proceduralAssignStmt) {
    return success(context.convertExpression(
        proceduralAssignStmt.as<slang::ast::ProceduralAssignStatement>()
            .assignment));
  }

  LogicalResult visit(const slang::ast::VariableDeclStatement &) {
    // TODO: not sure
    return success();
  }

  LogicalResult visit(const slang::ast::ExpressionStatement &exprStmt) {
    return success(context.convertExpression(
        exprStmt.as<slang::ast::ExpressionStatement>().expr));
  }

  LogicalResult visit(const slang::ast::StatementList &listStmt) {
    for (auto *stmt : listStmt.list) {
      auto succeeded = (*stmt).visit(*this);
      if (succeeded.failed())
        return succeeded;
    }
    return success();
  }

  LogicalResult visit(const slang::ast::BlockStatement &blockStmt) {
    Context::SymbolTableScopeT varScope(context.varSymbolTable);
    return blockStmt.body.visit(*this);
  }

  LogicalResult visit(const slang::ast::EmptyStatement &emptyStmt) {
    return success();
  }

  LogicalResult visit(const slang::ast::TimedStatement &timeStmt) {
    if (failed(context.visitTimingControl(
            &timeStmt.as<slang::ast::TimedStatement>().timing)))
      return failure();
    if (failed(timeStmt.stmt.visit(*this)))
      return failure();

    return success();
  }

  /// Handle Loop
  LogicalResult visit(const slang::ast::ForLoopStatement &forStmt) {
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
    mlir::SmallVector<mlir::Type> types;
    auto type = context.convertType(*forStmt.stopExpr->type, loc);

    auto whileOp = builder.create<mlir::scf::WhileOp>(
        loc, types, mlir::SmallVector<Value, 0>{});
    OpBuilder::InsertionGuard guard(builder);

    // The before-region of the WhileOp.
    Block *before = builder.createBlock(&whileOp.getBefore());
    builder.setInsertionPointToEnd(before);
    Value cond = context.convertExpression(*forStmt.stopExpr);
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

    auto succeeded = forStmt.body.visit(*this);
    //   step_stmt in forLoop
    for (auto *steps : forStmt.steps) {
      context.convertExpression(*steps);
    }
    builder.create<mlir::scf::YieldOp>(loc);
    return succeeded.success();
  }

  LogicalResult visit(const slang::ast::RepeatLoopStatement &repeatStmt) {
    auto type = context.convertType(*repeatStmt.count.type, loc);
    Value countExpr = context.convertExpression(repeatStmt.count);
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

    auto succeeded = repeatStmt.body.visit(*this);

    // count decrement
    auto one = builder.create<moore::ConstantOp>(loc, type, 1);
    auto count = after->getArgument(0);
    auto result = builder.create<moore::SubOp>(loc, count, one);

    builder.create<mlir::scf::YieldOp>(loc, result->getResults());

    return succeeded.success();
  }

  LogicalResult visit(const slang::ast::WhileLoopStatement &whileStmt) {
    mlir::SmallVector<mlir::Type> types;
    auto type = context.convertType(*whileStmt.cond.type, loc);

    auto whileOp = builder.create<mlir::scf::WhileOp>(
        loc, types, mlir::SmallVector<Value, 0>{});
    OpBuilder::InsertionGuard guard(builder);

    // The before-region of the WhileOp.
    Block *before = builder.createBlock(&whileOp.getBefore());
    builder.setInsertionPointToEnd(before);
    Value cond = context.convertExpression(whileStmt.cond);
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

    auto succeeded = whileStmt.body.visit(*this);
    builder.create<mlir::scf::YieldOp>(loc);
    return succeeded.success();
  }

  LogicalResult visit(const slang::ast::DoWhileLoopStatement &dowhileStmt) {
    mlir::SmallVector<mlir::Type> types;
    auto type = context.convertType(*dowhileStmt.cond.type, loc);

    auto whileOp = builder.create<mlir::scf::WhileOp>(
        loc, types, mlir::SmallVector<Value, 0>{});
    OpBuilder::InsertionGuard guard(builder);

    // The before-region of the WhileOp.
    Block *before = builder.createBlock(&whileOp.getBefore());
    builder.setInsertionPointToEnd(before);

    auto succeeded = dowhileStmt.body.visit(*this);
    Value cond = context.convertExpression(dowhileStmt.cond);
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

  /// Emit an error for all other statement.
  template <typename T>
  LogicalResult visit(T &&node) {
    mlir::emitError(loc, "unsupported statement: ")
        << slang::ast::toString(node.kind);
    return mlir::failure();
  }

  LogicalResult visitInvalid(const slang::ast::Statement &stmt) {
    mlir::emitError(loc, "invalid statement");
    return mlir::failure();
  }
};
} // namespace

// It can handle the statements like case, conditional(if), for loop, and etc.
LogicalResult
Context::convertStatement(const slang::ast::Statement *statement) {
  auto loc = convertLocation(statement->sourceRange.start());

  return (*statement).visit(StmtVisitor{*this, loc, builder});
}
