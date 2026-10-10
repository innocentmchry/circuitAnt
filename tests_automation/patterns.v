// One of each structural pattern (see test_engine.Patterns).
//   inverter pairs : g1->g2, g11->g12
//   duplicates     : g3 = and(a, b), g4 = and(b, a)
//   enable flop F1 : D = (en & d) | (~en & q)   -- merge at or-gate g10
//   hold flop   F2 : D = not(not(h))            -- direct hold
module top(clk, rst, a, b, en, d, y1, y2, q, h);
  input  clk, rst, a, b, en, d;
  output y1, y2, q, h;
  wire n1, n2, n3, n4, nen, sel_d, sel_q, m, hq1, hq2;

  not g1(n1, a);
  not g2(n2, n1);
  and g3(n3, a, b);
  and g4(n4, b, a);
  or  g5(y1, n3, n2);
  buf g6(y2, n4);
  not g7(nen, en);
  and g8(sel_d, en, d);
  and g9(sel_q, nen, q);
  or  g10(m, sel_d, sel_q);
  dff F1(.RN(rst), .SN(1'b1), .CK(clk), .D(m), .Q(q));
  not g11(hq1, h);
  not g12(hq2, hq1);
  dff F2(.RN(rst), .SN(1'b1), .CK(clk), .D(hq2), .Q(h));
endmodule
