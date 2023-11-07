// RUN: circt-translate --import-verilog %s | FileCheck %s

// CHECK-LABEL: moore.module @top {
module top();
  // CHECK-NEXT: %passign_a = moore.variable : !moore.logic
  // CHECK-LABEL: moore.initial {
  // CHECK-NEXT:    %3 = moore.mir.constant 2 : !moore.logic
  // CHECK-NEXT:    moore.mir.passign %passign_a, %3 : !moore.logic
  // CHECK-NEXT:  }
  logic passign_a;
initial begin
	passign_a <= 2;
end

  // CHECK-NEXT: %int_a = moore.variable : !moore.int
  // CHECK-NEXT: %0 = moore.mir.vardecl "int_b" = 2 : !moore.int
int int_a;
int int_b=2;
  // CHECK-NEXT: %1 = moore.mir.constant 3 : !moore.logic
  // CHECK-NEXT: %bpassign_a = moore.variable : !moore.logic
  // CHECK-NEXT: moore.mir.bpassign %bpassign_a, %1 : !moore.logic
logic bpassign_a = 3;
  // CHECK-NEXT: %2 = moore.mir.constant 2 : !moore.logic
  // CHECK-NEXT: %bpassign_b = moore.variable : !moore.logic
  // CHECK-NEXT: moore.mir.bpassign %bpassign_b, %2 : !moore.logic
logic bpassign_b = 2;

  // CHECK-LABEL: moore.initial {
  // CHECK-NEXT:    %3 = moore.mir.constant 1 : !moore.logic
  // CHECK-NEXT:    moore.mir.bpassign %bpassign_a, %3 : !moore.logic
  // CHECK-NEXT:    moore.mir.bpassign %bpassign_b, %3 : !moore.logic
  // CHECK-NEXT:    %4 = moore.mir.constant 2 : !moore.int
  // CHECK-NEXT:    moore.mir.bpassign %int_a, %4 : !moore.int
  // CHECK-NEXT:  }
initial begin
	bpassign_a = 1;
	bpassign_b = bpassign_a;
  assign int_a=2;
end

endmodule
