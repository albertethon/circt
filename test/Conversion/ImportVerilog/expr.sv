// RUN: circt-translate --import-verilog %s | FileCheck %s

// CHECK-LABEL: moore.module @top {
module top();
  // CHECK-NEXT: %passign_a = moore.variable : !moore.logic
  // CHECK-LABEL: moore.initial {
  // CHECK-NEXT:    %2 = moore.mir.constant 2 : !moore.logic
  // CHECK-NEXT:    moore.mir.passign %passign_a, %2 : !moore.logic
  // CHECK-NEXT:  }
  logic passign_a;
initial begin
	passign_a <= 2;
end

  // CHECK-NEXT: %int_a = moore.variable : !moore.int
  // CHECK-NEXT: %0 = moore.mir.constant 3 : !moore.logic
  // CHECK-NEXT: %1 = moore.mir.constant 2 : !moore.logic
int int_a;
logic bpassign_a = 3;
logic bpassign_b = 2;

  // CHECK-LABEL: moore.initial {
  // CHECK-NEXT:    %2 = moore.mir.constant 1 : !moore.logic
  // CHECK-NEXT:    moore.mir.bpassign %0, %2 : !moore.logic
  // CHECK-NEXT:    moore.mir.bpassign %1, %2 : !moore.logic
  // CHECK-NEXT:    %3 = moore.mir.constant 2 : !moore.int
  // CHECK-NEXT:    moore.mir.bpassign %int_a, %3 : !moore.int
  // CHECK-NEXT:  }
initial begin
	bpassign_a = 1;
	bpassign_b = bpassign_a;
  assign int_a=2;
end

endmodule
