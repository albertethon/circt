//===- Statement.cpp - Slang expression conversion-------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "ImportVerilogInternals.h"
#include "slang/ast/ASTVisitor.h"
#include "slang/ast/Expression.h"
#include "slang/ast/Statements.h"
#include "slang/ast/Symbol.h"
#include "slang/ast/symbols/CompilationUnitSymbols.h"
#include "slang/ast/symbols/InstanceSymbols.h"
#include "slang/ast/symbols/MemberSymbols.h"
#include "slang/ast/symbols/VariableSymbols.h"
#include "slang/ast/types/AllTypes.h"
#include "slang/ast/types/Type.h"
#include "slang/syntax/SyntaxVisitor.h"

using namespace circt;
using namespace ImportVerilog;

// Detail processing about integer literal.
Value Context::visitIntegerLiteral(
    const slang::ast::IntegerLiteral *integerLiteralExpr) {
  auto srcValue = integerLiteralExpr->getValue().as<uint32_t>().value();
  return rootBuilder.create<moore::ConstantOp>(
      convertLocation(integerLiteralExpr->sourceRange.start()),
      convertType(*integerLiteralExpr->type), srcValue);
}

// Detail processing about named value.
Value Context::visitNamedValue(
    const slang::ast::NamedValueExpression *namedValueExpr) {
    auto loc = convertLocation(namedValueExpr->sourceRange.start());
    llvm::StringRef varname(namedValueExpr->getSymbolReference()->name);
    auto [value,expression] = varSymbolTable.lookup(varname);
    if(value){
      return value;
    }
    mlir::emitError(loc, "error: unknow variable '")
      << namedValueExpr->getSymbolReference()->name << "'";
    return nullptr;
}

// Detail processing about procedural assignment.
Value Context::visitAssignmentExpr(
    const slang::ast::AssignmentExpression *assignmentExpr) {
  auto loc = convertLocation(assignmentExpr->sourceRange.start());
  Value lhs = visitExpression(&assignmentExpr->left());
  Value rhs;
  if (!lhs)
    return nullptr;
  // assign a=b;
  // check b is defined
  if(assignmentExpr->right().as_if<slang::ast::NamedValueExpression>()){
    auto rhs_varname = assignmentExpr->right().getSymbolReference()->name;
    auto rhs_pair = varSymbolTable.lookup(rhs_varname);
    rhs = rhs_pair.first;
    const slang::ast::Expression* rhs_expr = rhs_pair.second;

    if(rhs_expr==nullptr){
      //TODO: implicit declaration is not allowed
      mlir::emitError(loc, "error: undefined variable '")
      << rhs_varname << "'";
    }
  }
  else{
    rhs = visitExpression(&assignmentExpr->right());
    if (!rhs)
      return nullptr;
  }
  // assign the rhs's operation and expr to varname in lhs
  auto varname = assignmentExpr->left().getSymbolReference()->name;
  varSymbolTable.insert(varname,{rhs,&assignmentExpr->right()});

  if(assignmentExpr->isNonBlocking()){
    rootBuilder.create<moore::PAssignOp>(loc, lhs, rhs);
  }else{
    rootBuilder.create<moore::BPAssignOp>(loc, lhs, rhs);
  }
  return lhs;
}

// Detail processing about continuous assignment.
Value Context::visitContinuousAssignmentExpr(
    const slang::ast::AssignmentExpression *assignmentExpr){
  auto loc = convertLocation(assignmentExpr->sourceRange.start());
  Value lhs = visitExpression(&assignmentExpr->left());
  Value rhs;
  if (!lhs)
    return nullptr;
  // assign a=b;
  // check b is defined
  if(assignmentExpr->right().as_if<slang::ast::NamedValueExpression>()){
    auto rhs_varname = assignmentExpr->right().getSymbolReference()->name;
    auto rhs_pair = varSymbolTable.lookup(rhs_varname);
    rhs = rhs_pair.first;
    const slang::ast::Expression* rhs_expr = rhs_pair.second;

    if(rhs_expr==nullptr){
      //TODO: implicit declaration is not allowed
      mlir::emitError(loc, "error: undefined variable '")
      << rhs_varname << "'";
    }
  }
  else{
    rhs = visitExpression(&assignmentExpr->right());
    if (!rhs)
      return nullptr;
  }
  // assign the rhs's operation and expr to varname in lhs
  auto varname = assignmentExpr->left().getSymbolReference()->name;
  varSymbolTable.insert(varname,{rhs,&assignmentExpr->right()});

  rootBuilder.create<moore::AssignOp>(loc, lhs, rhs);
  return lhs;
    }

// Detail processing about conversion
Value Context::visitConversion(
    const slang::ast::ConversionExpression *conversionExpr,
    const slang::ast::Type &type) {
  auto loc = convertLocation(conversionExpr->sourceRange.start());
  switch (conversionExpr->operand().kind) {
  case slang::ast::ExpressionKind::IntegerLiteral:
    // For assignment, which formation of the right hand is
    // {coversion(logic){conversion(logic signed [31:0]){IntegerLiteral(int)}}}
    // to make sure the type is the same on both sides of the equation.
    return rootBuilder.create<moore::ConstantOp>(
        loc, convertType(type),
        conversionExpr->operand()
            .as<slang::ast::IntegerLiteral>()
            .getValue()
            .as<uint32_t>()
            .value());
  case slang::ast::ExpressionKind::NamedValue:
    return visitNamedValue(
        &conversionExpr->operand().as<slang::ast::NamedValueExpression>());
  case slang::ast::ExpressionKind::BinaryOp:
    mlir::emitError(loc, "unsupported conversion expression: binary operator");
    return nullptr;
  case slang::ast::ExpressionKind::ConditionalOp:
    mlir::emitError(loc,
                    "unsupported conversion expression: conditional operator");
    return nullptr;
  case slang::ast::ExpressionKind::Conversion:
    return visitConversion(
        &conversionExpr->operand().as<slang::ast::ConversionExpression>(),
        *conversionExpr->type);
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
Value Context::visitExpression(const slang::ast::Expression *expression) {
  auto loc = convertLocation(expression->sourceRange.start());
  switch (expression->kind) {
  case slang::ast::ExpressionKind::IntegerLiteral:
    return visitIntegerLiteral(&expression->as<slang::ast::IntegerLiteral>());
  case slang::ast::ExpressionKind::NamedValue:
    return visitNamedValue(&expression->as<slang::ast::NamedValueExpression>());
  case slang::ast::ExpressionKind::Assignment:
    return visitAssignmentExpr(
        &expression->as<slang::ast::AssignmentExpression>());
  case slang::ast::ExpressionKind::Conversion:
    return visitConversion(&expression->as<slang::ast::ConversionExpression>(),
                           *expression->type);
  // There is other cases.
  default:
    mlir::emitError(loc, "unsupported expression");
    return nullptr;
  }

  return nullptr;
}
