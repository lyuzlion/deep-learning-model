# 多头注意力

$Q = XW_q, \quad K = XW_k,\quad V = XW_v$

$Attention(Q, K, V) = softmax( \frac{QK^T}{\sqrt{d_k}} )V$
