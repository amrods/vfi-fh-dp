#! value function iteration for finite horizon dp model with continuous state
using Optim
using Distributions

include("nelder_mead.jl")

# transformation functions
# closed interval [a, b]
function tab(x; a = 0, b = 1)
    (b + a)/2 + (b - a)/2*((2x)/(1 + x^2))
end

# open interval (a, b)
function logit(x; a = 0, b = 1)
    (b - a) * (exp(x)/(1 + exp(x))) + a
end

const transf = tab

# function that solves last period, for now it uses Optim.jl
function solvelast!(dp::NamedTuple, Ldict::Array{Ty}, Cdict::Array{Ty}, A1dict::Array{Ty}, Vdict::Array{Ty}) where Ty <: Real
    u = dp.u
    grid_A = dp.grid_A
    n = dp.n
    w = dp.w
    ξ = dp.ξ
    r = dp.r
    T = dp.T
    for s in 1:length(grid_A)
        for i in 1:n
            opt = optimize( x -> -u((w[T] + ξ[i])*x + grid_A[s]*(1+r), x), 0.0, 1.0 )
            Ldict[i, s, T] = Optim.minimizer(opt)
            Cdict[i, s, T] = (w[T] + ξ[i])*Ldict[i, s, T] + grid_A[s]*(1+r)
            A1dict[i, s, T] = 0.0
            Vdict[i, s, T] = -Optim.minimum(opt)
            iterdict[i, s, T] = -1
        end
    end
    return Ldict, Cdict, A1dict, Vdict
end

# function that solves the rest of periods
function solverest!(dp::NamedTuple, Ldict::Array{Ty}, Cdict::Array{Ty}, A1dict::Array{Ty}, Vdict::Array{Ty}, iterdict::Array{Int}; t0::Int=1) where Ty <: Real
    u = dp.u
    grid_A = dp.grid_A
    n = dp.n
    w = dp.w
    ξ = dp.ξ
    r = dp.r
    T = dp.T

    for t in T-1:-1:t0
        # interpolation
        interp_func_t1 = LinearInterpolation(grid_A, sum(Vdict[i, :, t+1] for i in 1:n) / n, extrapolation_bc = Line())
        for s in 1:length(grid_A)
            for i in 1:n
                # x[1] is assets to carry forward, x[2] is labor supply
                if w[t] + grid_A[s] * (1+r) < 0
                    Vdict[i, s, t] = -Inf
                    Ldict[i, s, t] = NaN
                    A1dict[i, s, t] = NaN
                    Cdict[i, s, t] = NaN
                    iterdict[i, s, t] = -1
                else
                    initial_x = [[100.0, 0.0], [0.0, 0.0], [0.0, 100.0]]
                    opt = nelder_mead( x -> -(u(transf(x[2])*(w[t] + ξ[i])+ grid_A[s]*(1+r) - x[1], transf(x[2])) +
                            β*interp_func_t1(x[1]) ),
                            initial_x, 1e-8 )
                    Vdict[i, s, t] = -opt[1]
                    xstar = opt[2]
                    Ldict[i, s, t] = transf(xstar[2])
                    A1dict[i, s, t] = xstar[1]
                    Cdict[i, s, t] = Ldict[i, s, t] * (w[t] + ξ[i]) + grid_A[s] * (1+r) - A1dict[i, s, t]
                    iterdict[i, s, t] = opt[3]
                end
            end
        end
        println("period ", t, " finished")
    end
    return Ldict, Cdict, A1dict, Vdict, iterdict
end

# function that solves the all periods
function solvemodel!(dp::NamedTuple, Ldict::Array{Ty}, Cdict::Array{Ty}, A1dict::Array{Ty}, Vdict::Array{Ty}, iterdict::Array{Int}; t0::Int=1) where Ty <: Real
    solvelast!(dp, Ldict, Cdict, A1dict, Vdict)
    solverest!(dp, Ldict, Cdict, A1dict, Vdict, iterdict; t0=t0)
    return Ldict, Cdict, A1dict, Vdict, iterdict
end

# instantiate model
function u(c::R, L::R) where R <: Real
    if c <= 0 || !(0 <= 1 - L <= 1)
        return -Inf
    else
        return log(c) + log(1 - L)
    end
end

T = 65 # terminal period
β = 0.95 # discount factor
r = 0.05 # interest rate
dist = LogNormal(0, 1) # distribution of ξ
n = 5 # number of points of support of ξ

w = Vector{Float64}(undef, T) # exogenous wages
w .= (900 .+ 20.0 .* (1:T) .- 0.5 .* (1:T).^2)

ξ = quantile.(dist, 0:1/n:prevfloat(1.0))

grid_A = -800:10.0:8_000

Vdict = Array{Float64}(undef, (n, length(grid_A), T))
Cdict = similar(Vdict)
Ldict = similar(Vdict)
A1dict = similar(Vdict)
iterdict = zeros(Int, n, length(grid_A), T)

dp = (u=u, T=T, β=β, r=r, n=n, w=w, grid_A=grid_A, ξ=ξ)

@time solvemodel!(dp, Ldict, Cdict, A1dict, Vdict, iterdict)
