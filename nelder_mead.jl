#! algorithm 7.7 of Kochenderfer & Wheeler (2019) with some modifications
using StatsBase: mean, std

# nelder_mead is a minimization routine
# i also implement a slight generalization of the algorithm in KW, allowing for a flexible shrinkage parameter
# f must be an anonymous function that takes a vector as argument, ie x -> x[1]^2 + x[2]^2
# S must be a n+1 simplex, specified as a vector of vectors, ie [[0.0, 0.0], [1.0, 0.0], [1.0, 1.0]]
# There are no checks for whether S forms a proper simplex
# ε controls the tolerance of convergence
# α, β, γ, and δ control the reflection, expansion, contraction and shrinkage steps
# Gao & Han (2010) suggest an adaptive Nelder Mead algorithm by setting the parameters as follow:
# α = 1, β = 1 + 2/n, γ = 0.75 - 1/(2n), δ = 1 - 1/n, where n is the dimension of x

function g_nelder_mead(f, S, ε; α=1.0, β=2.0, γ=0.5, δ=0.5)
    Δ, y_arr = Inf, f.(S)
    iter = 0
    while Δ > ε
        iter += 1
        p = sortperm(y_arr) # sort lowest to highest
        S, y_arr = S[p], y_arr[p]
        xl, yl = S[1], y_arr[1] # lowest
        xh, yh = S[end], y_arr[end] # highest􏵅􏴙􏳴􏵅􏴚􏴅􏳳
        xs, ys = S[end-1], y_arr[end-1] # second-highest
        xm = mean(S[1:end-1]) # centroid
        xr = xm + α*(xm - xh) # reflection point
        yr = f(xr)

        if yr < yl
            xe = xm + β*(xr - xm) # expansion point
            ye = f(xe)
            S[end],y_arr[end] = ye < yr ? (xe, ye) : (xr, yr)
        elseif yr > ys
            if yr ≤ yh
                xh, yh, S[end], y_arr[end] = xr, yr, xr, yr
            end
            xc = xm + γ*(xh - xm) # contraction point
            yc = f(xc)
            if yc > yh
                for i in 2 : length(y_arr)
                    S[i] = xl + δ*(S[i] - xl) # some times this is parameterized as well with δ
                    y_arr[i] = f(S[i])
                end
            else
                S[end], y_arr[end] = xc, yc
            end
        else
            S[end], y_arr[end] = xr, yr
        end

        Δ = std(y_arr, corrected=false)
    end
    r = S[argmin(y_arr)]
    return f(r), r, iter
end

function nelder_mead(f, S, ε; α=1.0, β=2.0, γ=0.5)
    g_nelder_mead(f, S, ε; α=1.0, β=2.0, γ=0.5, δ=0.5)
end
