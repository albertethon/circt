
#include "ImportVerilogInternals.h"
#include "mlir/IR/Value.h"
#include "mlir/Support/LogicalResult.h"
#include "slang/ast/ASTVisitor.h"
#include <slang/ast/expressions/MiscExpressions.h>
#include <slang/ast/symbols/VariableSymbols.h>

using namespace circt;
using namespace ImportVerilog;


namespace {
struct ExprVisitor {
  Context &context;
  Location loc;
  ExprVisitor(Context &context, Location loc) : context(context), loc(loc), builder(context.rootBuilder) {}
  mlir::OpBuilder builder;
  Value visit(const slang::ast::IntegerLiteral &expr,const slang::ast::VariableSymbol &var){
    Type tmpType = context.convertType(*var.getDeclaredType());
    uint32_t value = expr.getValue().as<uint32_t>().value();
    return builder.create<moore::VariableDeclOp>(loc,
                    moore::LValueType::get(tmpType),
                    builder.getStringAttr(var.name),
                    value);
  }



  

 /// Emit an error for all other types.
  template<typename T, typename... Args>
  Value visit(T&& node, Args&&... args) {
    mlir::emitError(loc, "unsupported expr: ");
    return nullptr;
  }
  /// Emit an error for all other types.
  template<typename T, typename... Args>
  Value visitInvalid(T&& node, Args&&... args) {
    mlir::emitError(loc, "unsupported expr: ");
    return nullptr;
  }
};
} // namespace

void Context::convertExpr(const slang::ast::VariableSymbol &varAst, LocationAttr loc) {
  auto *expr = varAst.getInitializer();
  if (!loc)
    loc = convertLocation(expr->sourceRange.start());
  auto value = expr->visit(ExprVisitor(*this, loc),varAst);

}
