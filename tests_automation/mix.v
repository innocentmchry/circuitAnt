// Nangate45 cells ABC must handle: multi-output FA/HA, MUX2, AOI21, a QN-only flop
// (see test_engine.GenericVerification).
module mix (a, b, c, s, clk, y, co, h, m, q);
  input a, b, c, s, clk;
  output y, co, h, m, q;
  wire sum, hc, aoi, qn, nq;

  FA_X1 fa ( .A(a), .B(b), .CI(c), .CO(co), .S(sum) );
  HA_X1 ha ( .A(sum), .B(c), .CO(hc), .S(h) );
  MUX2_X1 mx ( .A(hc), .B(a), .S(s), .Z(m) );
  AOI21_X1 ao ( .A(a), .B1(b), .B2(nq), .ZN(aoi) );
  DFF_X1 f1 ( .D(aoi), .CK(clk), .QN(qn) );
  INV_X1 i1 ( .A(qn), .ZN(nq) );
  XOR2_X1 x1 ( .A(nq), .B(sum), .Z(y) );
  BUF_X1 b1 ( .A(nq), .Z(q) );
endmodule
