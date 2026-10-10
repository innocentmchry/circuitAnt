// Cases for the transformations (see test_engine.Transformations).
//   constants   : g1 = nand(a, 1) -> not(a);  g2 = and(b, 0) is a constant: skipped
//   dangling    : g4 -> g5 feed nothing
//   port to port: g6 -> g7 is an inverter pair from input a to output y3
module top(a, b, c, y1, y2, y3);
  input  a, b, c;
  output y1, y2, y3;
  wire n1, n2, d1, d2;

  nand g1(y1, a, 1'b1);
  and  g2(n1, b, 1'b0);
  or   g3(y2, n1, c);
  not  g4(d1, a);
  and  g5(d2, d1, b);
  not  g6(n2, a);
  not  g7(y3, n2);
endmodule
