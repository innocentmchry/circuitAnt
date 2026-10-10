module top(clk, rst, a, b, y, q);
  input  clk, rst, a, b;
  output y, q;
  wire n1, n2;

  and g1(n1, a, b);
  not g2(n2, n1);
  or  g3(y,  n2, a);
  dff g4(.RN(rst), .SN(1'b1), .CK(clk), .D(y), .Q(q));
endmodule
