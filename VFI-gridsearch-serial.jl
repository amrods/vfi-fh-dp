
using Parameters
using QuantEcon: rouwenhorst
using Interpolations

function solvelast!(dp::NamedTuple, Ldict, Cdict, A1dict, Vdict)
    utility = dp.utility
    grid_A = dp.grid_A
    n = dp.n
    w = dp.w
    r = dp.r
    T = dp.T
    β = dp.β
    ρ = dp.ρ
    σ = dp.σ
    μ = dp.μ

    # discretize ar(1) process in wages: rouwenhorst(n, ρ, σ, μ)
    mc = rouwenhorst(n, ρ, σ, μ)
    ξ = mc.state_values
    ℙ = mc.p

    for i in 1:n
        for s in 1:length(grid_A)
            vstar = -Inf
            Lstar = -Inf
            for L in 0:0.01:1
                v = utility(L*(w[T] + ξ[i]) + grid_A[s]*(1+r), L)
                if v >= vstar
                    vstar = v
                    Lstar = L
                end
            end
            Ldict[s, i, T] = Lstar
            Cdict[s, i, T] = (w[T] + ξ[i])*Ldict[s, i, T] + grid_A[s]*(1+r)
            A1dict[s, i, T] = 0.0
            Vdict[s, i, T] = vstar
        end
    end
    return Ldict, Cdict, A1dict, Vdict
end

function solverest!(dp::NamedTuple, Ldict, Cdict, A1dict, Vdict; t0::Int=1)
    utility = dp.utility
    grid_A = dp.grid_A
    n = dp.n
    w = dp.w
    r = dp.r
    T = dp.T
    β = dp.β
    ρ = dp.ρ
    σ = dp.σ
    μ = dp.μ

    # discretize ar(1) process in wages: rouwenhorst(n, ρ, σ, μ)
    mc = rouwenhorst(n, ρ, σ, μ)
    ξ = mc.state_values
    ℙ = mc.p

    for t in T-1:-1:t0
        for i in 1:n
            EV = sum(ℙ[i, i′] .* Vdict[:, i′, t+1] for i′ in 1:n)
            for s in 1:length(grid_A)
                vstar = -Inf
                Lstar = -Inf
                a1star = -Inf
                for a1 in 1:length(grid_A), L in 0:0.01:1
                    v = utility(L*(w[t] + ξ[i]) + grid_A[s]*(1+r) - grid_A[a1], L) + β*EV[a1]
                    if v >= vstar
                        vstar = v
                        Lstar = L
                        a1star = grid_A[a1]
                    end
                end
                Vdict[s, i, t] = vstar
                Ldict[s, i, t] = Lstar
                A1dict[s, i, t] = a1star
                Cdict[s, i, t] = Ldict[s, i, t] * (w[t] + ξ[i]) + grid_A[s] * (1+r) - A1dict[s, i, t]
            end
        end
        println("period ", t, " finished")
    end
    return Ldict, Cdict, A1dict, Vdict
end

function solvemodel!(dp::NamedTuple, Ldict, Cdict, A1dict, Vdict; t0::Int=1)
    solvelast!(dp, Ldict, Cdict, A1dict, Vdict)
    solverest!(dp, Ldict, Cdict, A1dict, Vdict; t0=t0)
    return Ldict, Cdict, A1dict, Vdict
end

function utility(c, L)
    if c <= 0 || 1 - L <= 0
        return -1e9
    else
        return log(c) + log(1 - L)
    end
end

T = 65
w = Vector{Float64}(undef, T) # exogenous wages
w .= (900 .+ 20.0 .* (1:T) .- 0.5 .* (1:T).^2)

# create model object with default values for some parameters
Model = @with_kw (utility=utility, n=15, w, r=0.05, T=65, β=0.95,
                                grid_A=-1_000:10.0:10_000, ρ=0.9, σ=0.8, μ=0.0)

dp = Model(w=w)

Vdict = Array{Float64}(undef, (length(dp.grid_A), dp.n, dp.T))
Cdict = Array{Float64}(undef, (length(dp.grid_A), dp.n, dp.T))
Ldict = Array{Float64}(undef, (length(dp.grid_A), dp.n, dp.T))
A1dict = Array{Float64}(undef, (length(dp.grid_A), dp.n, dp.T))

#@time solvemodel!(dp, Ldict, Cdict, A1dict, Vdict);
