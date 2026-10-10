// Two flip-flops, for depth across endpoint classes (see test_engine.Classes).
//   a,b -> g1(and) -> n1 -> g2(not) -> y          F1: D = n3, Q = q1
//          n1, q2  -> g3(or)  -> n3               F2: D = q1, Q = q2
module top(clk, rst, a, b, y);
  input  clk, rst, a, b;
  output y;
  wire n1, n3, q1, q2;

  and g1(n1, a, b);
  not g2(y, n1);
  or  g3(n3, n1, q2);
  dff F1(.RN(rst), .SN(1'b1), .CK(clk), .D(n3), .Q(q1));
  dff F2(.RN(rst), .SN(1'b1), .CK(clk), .D(q1), .Q(q2));
endmodule
