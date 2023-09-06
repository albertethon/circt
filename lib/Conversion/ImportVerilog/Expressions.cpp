
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

  //XXX:
  Value visit(const slang::ast::NamedValueExpression &expr,const slang::ast::VariableSymbol &var){
    llvm::StringRef sr(expr.getSymbolReference()->name);
    if(auto val = context.varSymbolTable.lookup(sr)){
      return val;
    }
    mlir::emitError(loc, "error: unknown variable '")
        << expr.getSymbolReference()->name << "'";
    return nullptr;
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
  if(!value)
    return;
  varSymbolTable.insert(varAst.name, value);

}
