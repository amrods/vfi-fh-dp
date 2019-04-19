
using SharedArrays
using Distributed

np = minimum([Sys.CPU_THREADS, 8])
addprocs(np - 1)

@everywhere using Parameters
@everywhere using QuantEcon: rouwenhorst
using LinearAlgebra

function solvelast!(dp::NamedTuple, Ldict, Cdict, A1dict, Vdict)
    utility = dp.u
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

    @sync @distributed for i in 1:n
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
    utility = dp.u
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
        EV = transpose( ℙ * transpose(Vdict[:, :, t+1]) )
        @time @sync @distributed for i in 1:n
            for s in 1:length(grid_A)
                vstar = -Inf
                Lstar = -Inf
                a1star = -Inf
                for a1 in 1:length(grid_A), L in 0:0.01:1
                    v = utility(L*(w[t] + ξ[i]) + grid_A[s]*(1+r) - grid_A[a1], L) +
                        β*EV[a1]
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
    @time solvelast!(dp, Ldict, Cdict, A1dict, Vdict)
    @time solverest!(dp, Ldict, Cdict, A1dict, Vdict; t0=t0)
    return Ldict, Cdict, A1dict, Vdict
end

@everywhere function utility(c, L)
    if c <= 0 || 1 - L <= 0
        return -1e6
    else
        return log(c) + log(1 - L)
    end
end

@everywhere T = 65
@everywhere w = Vector{Float64}(undef, T) # exogenous wages
w .= (900 .+ 20.0 .* (1:T) .- 0.5 .* (1:T).^2)

# create model object with default values for some parameters
@everywhere Model = @with_kw (u=utility, n=5, w, r=0.05, T=65, β=0.95, grid_A=-1_000:10.0:10_000, ρ=0.7, σ=15.0, μ=0.0)

@everywhere dp = Model(w=w)

Vdict = SharedArray{Float64}((length(dp.grid_A), dp.n, dp.T))
Cdict = SharedArray{Float64}((length(dp.grid_A), dp.n, dp.T))
Ldict = SharedArray{Float64}((length(dp.grid_A), dp.n, dp.T))
A1dict = SharedArray{Float64}((length(dp.grid_A), dp.n, dp.T))

@time solvemodel!(dp, Ldict, Cdict, A1dict, Vdict);
