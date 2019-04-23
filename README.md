# vfi-fh-dp
value function iteration for a simple finite horizon dp model:

$$
\max_{c_t, L_t, a_{t+1}} \mathbb{E}_t \left[ \sum_{n=0}^{T-t} \beta^n u(c_{t+n}, L_{t+n}) \right] \\
\text{s.t. } c_t + a_{t+1} = w_t(\xi_t) L_t + a_t(1+r), \text{ with }a_{0} \text{ given, and } a_T \geq 0.
$$

where
* $c_t$: consumption
* $L_t$: labor supply
* $a_t$: asset holdings after all actions have been completed in period $t$
* $w_t$: wage rate
    * $\xi_t$ is the random component of wages
* $r$: interest rate

Equivalently,
$$
\max_{L_t, a_{t+1}} \mathbb{E}_t \left[ \sum_{t=0}^T \beta^t u(w_t(\xi_t) L_t + a_t(1+r) - a_{t+1}, L_t) \right].
$$

The Bellman equation is
$$
V_t(a_t, \xi_t) = \max_{L_t, a_{t+1}} u(w_t(\xi_t) L_t + a_t(1+r) - a_{t+1}, L_t) + \beta \mathbb{E}_t \left[ V_{t+1}(a_{t+1}, \xi_{t+1}) \right]
$$

We assume the following functional forms
$$
u(c_t, L_t) = \log(c_t) + \log(1 - L_t)
$$

At each period $t$, the individual receives a wage offer $w_t$
$$
w_t(\xi_t) = \bar{w}_t + \xi_t
$$
where,
* $\bar{w}_t$ is a function of age
* stochastic component $\xi_t$ follows an AR(1) process
$$
\xi_t = \pi \xi_{t-1} + \varepsilon_t \\
\log \varepsilon_t \sim \mathcal{N}(0, \sigma_\varepsilon^2)
$$
