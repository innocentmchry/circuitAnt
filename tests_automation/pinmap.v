// Tiny positional pin-order check for the ICCAD cells (see test_engine.PinOrder).
module top(a, b, ck, rn, sn, d, y_and, y_not, q);
  input a, b, ck, rn, sn, d;
  output y_and, y_not, q;
  and g_and(y_and, a, b);
  not g_not(y_not, a);
  dff g_dff(ck, rn, sn, d, q);
endmodule
