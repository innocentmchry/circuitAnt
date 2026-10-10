// A Nangate45 tie cell driving a gate input (see test_engine.Generic).
module tiny(a, y);
  input  a;
  output y;
  wire one;

  LOGIC1_X1 t1 ( .Z(one) );
  AND2_X1 g1 ( .A1(a), .A2(one), .ZN(y) );
endmodule
