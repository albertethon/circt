//===- Statement.cpp - Slang expression conversion-------------------------===//
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

// Detail processing about integer literal.
Value Context::visitIntegerLiteral(
    const slang::ast::IntegerLiteral *integerLiteralExpr,
    const slang::ast::Type &type) {
  auto srcValue = integerLiteralExpr->getValue().as<uint32_t>().value();
  return rootBuilder.create<moore::ConstantOp>(
      convertLocation(integerLiteralExpr->sourceRange.start()),
      convertType(type), srcValue);
}

// Detail processing about named value.
Value Context::visitNamedValue(
    const slang::ast::NamedValueExpression *namedValueExpr,
    const slang::ast::Type &type) {
  // TODO:
  auto loc = convertLocation(namedValueExpr->getSymbolReference()->location);
  auto name = namedValueExpr->getSymbolReference()->name;

  // return varSymbolTable.lookup(name).first;
  return rootBuilder.create<moore::VariableOp>(loc, convertType(type), name);
  // return nullptr;
}

// Detail processing about binary expression.
Value Context::visitBinaryOp(const slang::ast::BinaryExpression *binaryExpr,
                             const slang::ast::Type &type) {
  auto loc = convertLocation(binaryExpr->sourceRange.start());
  Value lhs = visitExpression(&binaryExpr->left(), type);
  if (!lhs)
    return nullptr;
  Value rhs = visitExpression(&binaryExpr->right(), type);
  if (!rhs)
    return nullptr;

  switch (binaryExpr->op) {
  case slang::ast::BinaryOperator::Add:
    mlir::emitError(loc, "unsupported binary operator : add");
    return nullptr;
  case slang::ast::BinaryOperator::Subtract:
    mlir::emitError(loc, "unsupported binary operator : subtract");
    return nullptr;
  case slang::ast::BinaryOperator::Multiply:
    mlir::emitError(loc, "unsupported binary operator : multiply");
    return nullptr;
  case slang::ast::BinaryOperator::Divide:
    mlir::emitError(loc, "unsupported binary operator : divide");
    return nullptr;
  case slang::ast::BinaryOperator::Mod:
    mlir::emitError(loc, "unsupported binary operator : mod");
    return nullptr;
  case slang::ast::BinaryOperator::BinaryAnd:
    mlir::emitError(loc, "unsupported binary operator : binary and");
    return nullptr;
  case slang::ast::BinaryOperator::BinaryOr:
    mlir::emitError(loc, "unsupported binary operator : binary or");
    return nullptr;
  case slang::ast::BinaryOperator::BinaryXor:
    mlir::emitError(loc, "unsupported binary operator : binary xor");
    return nullptr;
  case slang::ast::BinaryOperator::BinaryXnor:
    mlir::emitError(loc, "unsupported binary operator : binary xnor");
    return nullptr;
  case slang::ast::BinaryOperator::Equality:
    return rootBuilder.create<moore::EqualityOp>(loc, lhs, rhs);
  case slang::ast::BinaryOperator::Inequality:
    return rootBuilder.create<moore::InEqualityOp>(loc, lhs, rhs);
  case slang::ast::BinaryOperator::CaseEquality:
    return rootBuilder.create<moore::EqualityOp>(loc, lhs, rhs,
                                                 rootBuilder.getUnitAttr());
  case slang::ast::BinaryOperator::CaseInequality:
    return rootBuilder.create<moore::InEqualityOp>(loc, lhs, rhs,
                                                   rootBuilder.getUnitAttr());
  case slang::ast::BinaryOperator::GreaterThanEqual:
    mlir::emitError(loc, "unsupported binary operator : greater than equal");
    return nullptr;
  case slang::ast::BinaryOperator::GreaterThan:
    mlir::emitError(loc, "unsupported binary operator : greater than");
    return nullptr;
  case slang::ast::BinaryOperator::LessThanEqual:
    mlir::emitError(loc, "unsupported binary operator : less than equal");
    return nullptr;
  case slang::ast::BinaryOperator::LessThan:
    mlir::emitError(loc, "unsupported binary operator : less than");
    return nullptr;
  case slang::ast::BinaryOperator::WildcardEquality:
    mlir::emitError(loc, "unsupported binary operator : wildcard equality");
    return nullptr;
  case slang::ast::BinaryOperator::WildcardInequality:
    mlir::emitError(loc, "unsupported binary operator : wildcard inequality");
    return nullptr;
  case slang::ast::BinaryOperator::LogicalAnd:
    mlir::emitError(loc, "unsupported binary operator : logical and");
    return nullptr;
  case slang::ast::BinaryOperator::LogicalOr:
    mlir::emitError(loc, "unsupported binary operator : logical or");
    return nullptr;
  case slang::ast::BinaryOperator::LogicalImplication:
    mlir::emitError(loc, "unsupported binary operator : logical implication");
    return nullptr;
  case slang::ast::BinaryOperator::LogicalEquivalence:
    mlir::emitError(loc, "unsupported binary operator : logical equivalence");
    return nullptr;
  case slang::ast::BinaryOperator::LogicalShiftLeft:
    mlir::emitError(loc, "unsupported binary operator : logical shift left");
    return nullptr;
  case slang::ast::BinaryOperator::LogicalShiftRight:
    mlir::emitError(loc, "unsupported binary operator : logical shift right");
    return nullptr;
  case slang::ast::BinaryOperator::ArithmeticShiftLeft:
    mlir::emitError(loc, "unsupported binary operator : arithmetic shift left");
    return nullptr;
  case slang::ast::BinaryOperator::ArithmeticShiftRight:
    mlir::emitError(loc,
                    "unsupported binary operator : arithmetic shift right");
    return nullptr;
  case slang::ast::BinaryOperator::Power:
    mlir::emitError(loc, "unsupported binary operator : power");
    return nullptr;
  default:
    mlir::emitError(loc, "unsupported binary operator");
    return nullptr;
  }
  return nullptr;
}

// Detail processing about assignment.
Value Context::visitAssignmentExpr(
    const slang::ast::AssignmentExpression *assignmentExpr,
    const slang::ast::Type &type) {
  auto loc = convertLocation(assignmentExpr->sourceRange.start());
  Value lhs = visitExpression(&assignmentExpr->left(), type);
  if (!lhs)
    return nullptr;
  Value rhs = visitExpression(&assignmentExpr->right(), type);
  if (!rhs)
    return nullptr;
  // TODO:
  // if (assignmentExpr->right().as_if<slang::ast::NamedValueExpression>()) {
  //   mlir::emitError(loc, "unsupported assignment of kind like a = b");
  //   return nullptr;
  // }
  // if (assignmentExpr->isNonBlocking())
  //   rootBuilder.create<moore::PAssignOp>(loc, lhs, rhs);
  // else {
  //   if (assignmentExpr->syntax->parent->kind ==
  //       slang::syntax::SyntaxKind::ContinuousAssign)
  //     rootBuilder.create<moore::AssignOp>(loc, lhs, rhs);
  //   else
  rootBuilder.create<moore::BPAssignOp>(loc, lhs, rhs);
  // }
  return lhs;
}

// Detail processing about conversion
Value Context::visitConversion(
    const slang::ast::ConversionExpression *conversionExpr,
    const slang::ast::Type &type) {
  auto loc = convertLocation(conversionExpr->sourceRange.start());
  switch (conversionExpr->operand().kind) {
  case slang::ast::ExpressionKind::IntegerLiteral:
    return visitIntegerLiteral(
        &conversionExpr->operand().as<slang::ast::IntegerLiteral>(), type);
  case slang::ast::ExpressionKind::NamedValue:
    return visitNamedValue(
        &conversionExpr->operand().as<slang::ast::NamedValueExpression>(),
        type);
  case slang::ast::ExpressionKind::BinaryOp:
    return visitBinaryOp(
        &conversionExpr->operand().as<slang::ast::BinaryExpression>(), type);
  case slang::ast::ExpressionKind::ConditionalOp:
    mlir::emitError(loc,
                    "unsupported conversion expression: conditional operator");
    return nullptr;
  case slang::ast::ExpressionKind::Conversion:
    return visitConversion(
        &conversionExpr->operand().as<slang::ast::ConversionExpression>(),
        type);
  case slang::ast::ExpressionKind::LValueReference:
    mlir::emitError(loc, "unsupported conversion expression: lValue reference");
    return nullptr;
  // There is other cases.
  default:
    mlir::emitError(loc, "unsupported conversion expression");
    return nullptr;
  }

  return nullptr;
}

// It can handle the expressions like literal, assignment, conversion, and etc,
// which can be reviewed in slang/include/slang/ast/ASTVisitor.h.
Value Context::visitExpression(const slang::ast::Expression *expression,
                               const slang::ast::Type &type) {
  auto loc = convertLocation(expression->sourceRange.start());
  switch (expression->kind) {
  case slang::ast::ExpressionKind::IntegerLiteral:
    return visitIntegerLiteral(&expression->as<slang::ast::IntegerLiteral>(),
                               type);
  case slang::ast::ExpressionKind::NamedValue:
    return visitNamedValue(&expression->as<slang::ast::NamedValueExpression>(),
                           type);
  case slang::ast::ExpressionKind::BinaryOp:
    return visitBinaryOp(&expression->as<slang::ast::BinaryExpression>(), type);
  case slang::ast::ExpressionKind::Assignment:
    return visitAssignmentExpr(
        &expression->as<slang::ast::AssignmentExpression>(), type);
  case slang::ast::ExpressionKind::Conversion:
    return visitConversion(&expression->as<slang::ast::ConversionExpression>(),
                           type);
  // There is other cases.
  default:
    mlir::emitError(loc, "unsupported expression");
    return nullptr;
  }

  return nullptr;
}
