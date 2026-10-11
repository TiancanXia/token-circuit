# Attention formulas

The query is at position $t$. Query head $h$ reads key/value head $g(h)$.
RoPE has already been applied to $\mathbf q$ and $\mathbf k$.

## Attention scores

$$
s_{tj}^{(h)} = \frac{\mathbf q_t^{(h)} \cdot \mathbf k_j^{(g(h))}}{\sqrt{d_h}}
$$

## Causal probabilities

$$
p_{tj}^{(h)} = \begin{cases}
\displaystyle \frac{\exp(s_{tj}^{(h)})}{\sum_{\ell=0}^{t}\exp(s_{t\ell}^{(h)})} & j \le t, \\[6pt]
0 & j > t.
\end{cases}
$$

## Weighted sum and output projection

$$
\begin{aligned}
\mathbf o_t^{(h)} &= \sum_{j=0}^{t} p_{tj}^{(h)}\mathbf v_j^{(g(h))} \\[6pt]
\mathbf y_t &= \operatorname{Concat}(\mathbf o_t^{(1)},\ldots,\mathbf o_t^{(H)})W_o^{\top}
\end{aligned}
$$

$d_h$ is the width of a head; $H$ is the number of query heads. In MHA,
$g(h)=h$. GQA maps several query heads to each shared key/value head.
