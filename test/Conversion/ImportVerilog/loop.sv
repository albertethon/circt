// RUN: circt-translate --import-verilog %s | FileCheck %s


// CHECK-LABEL: moore.module @while_tb
module while_tb ();
    static int i = 0;
	initial begin
	// CHECK: scf.while : () -> () {
	// CHECK:   [[TMP1:%.+]] = moore.constant 2 : !moore.int
	// CHECK:   [[TMP2:%.+]] = moore.ne %i, [[TMP1]] : !moore.int -> !moore.bit
	// CHECK:   [[TMP3:%.+]] = moore.constant false : !moore.bit
	// CHECK:   [[TMP4:%.+]] = moore.mir.ne [[TMP2]], [[TMP3]] : !moore.bit, !moore.bit
	// CHECK:   scf.condition([[TMP4]])
	// CHECK: } do {
	// CHECK:   [[TMP1:%.+]] = moore.constant 1 : !moore.int
	// CHECK:   [[TMP2:%.+]] = moore.add %i, [[TMP1]] : !moore.int
	// CHECK:   moore.mir.bpassign %i, [[TMP2]] : !moore.int
	// CHECK:   scf.yield
	// CHECK: }
		while(i != 2)begin
			i++;
		end
	end
endmodule

// CHECK-LABEL: moore.module @dowhile_tb
module dowhile_tb ();
    static int i = 0;
	initial begin
	// CHECK: scf.while : () -> () {
	// CHECK:   [[TMP1:%.+]] = moore.constant 1 : !moore.int
	// CHECK:   [[TMP2:%.+]] = moore.add %i, [[TMP1]] : !moore.int
	// CHECK:   moore.mir.bpassign %i, [[TMP2]] : !moore.int
	// CHECK:   [[TMP3:%.+]] = moore.constant 2 : !moore.int
	// CHECK:   [[TMP4:%.+]] = moore.ne %i, [[TMP3]] : !moore.int -> !moore.bit
	// CHECK:   [[TMP5:%.+]] = moore.constant false : !moore.bit
	// CHECK:   [[TMP6:%.+]] = moore.mir.ne [[TMP4]], [[TMP5]] : !moore.bit, !moore.bit
	// CHECK:   scf.condition([[TMP6]])
	// CHECK: } do {
	// CHECK:   scf.yield
	// CHECK: }
		do begin
			i++;
		end while(i != 2); 
	end
endmodule
