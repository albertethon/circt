
#include "ImportVerilogInternals.h"
#include "slang/ast/ASTVisitor.h"

using namespace circt;
using namespace ImportVerilog;


namespace {
struct ExprVisitor {
  Context &context;
  Location loc;
  ExprVisitor(Context &context, Location loc) : context(context), loc(loc), builder(context.rootBuilder) {}
  mlir::OpBuilder builder;

  //----------------------------------------------------------------------------------
  //Literal Expressions visit
  //----------------------------------------------------------------------------------

  ///XXX: visit IntegerLiteral and build constantOp, VariableDeclOp
  Value visit(const slang::ast::IntegerLiteral &expr,const slang::ast::VariableSymbol &var){
    Type tmpType = context.convertType(*var.getDeclaredType());
    uint32_t value = expr.getValue().as<uint32_t>().value();
    // if declare a constant type
    if(var.flags.has(slang::ast::VariableFlags::Const)){
      return builder.create<moore::ConstantOp>(loc,
                    tmpType,
                    value);
    }
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
