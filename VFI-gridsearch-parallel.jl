
using SharedArrays
using Distributed

np = minimum([Sys.CPU_THREADS, 8])
addprocs(np - 1)

@everywhere using Distributions

function solvelast!(dp::NamedTuple, Ldict, Cdict, A1dict, Vdict) #where TA
    utility = dp.u
    grid_A = dp.grid_A
    n = dp.n
    w = dp.w
    ξ = dp.ξ
    r = dp.r
    T = dp.T

    @sync @distributed for s in 1:length(grid_A)
        for i in 1:n
            vstar = -Inf
            Lstar = -Inf
            for L in 0:0.01:1
                v = utility(L*(w[T] + ξ[i]) + grid_A[s]*(1+r), L)
                if v > vstar
                    vstar = v
                    Lstar = L
                end
            end
            Ldict[i, s, T] = Lstar
            Cdict[i, s, T] = (w[T] + ξ[i])*Ldict[i, s, T] + grid_A[s]*(1+r)
            A1dict[i, s, T] = 0.0
            Vdict[i, s, T] = vstar
        end
    end
    return Ldict, Cdict, A1dict, Vdict
end

function solverest!(dp::NamedTuple, Ldict, Cdict, A1dict, Vdict; t0::Int=1) #where TA #<: AbstractArray
    utility = dp.u
    grid_A = dp.grid_A
    n = dp.n
    w = dp.w
    ξ = dp.ξ
    r = dp.r
    T = dp.T

    for t in T-1:-1:t0
        Ev = sum(Vdict[j, :, t+1] for j in 1:n)/n
        @time @sync @distributed for s in 1:length(grid_A)
            for i in 1:n
                # x[1] is assets to carry forward, x[2] is labor supply
                vstar = -Inf
                Lstar = -Inf
                a1star = -Inf
                for a1 in 1:length(grid_A), L in 0:0.01:1
                    v = utility(L*(w[t] + ξ[i]) + grid_A[s]*(1+r) - grid_A[a1], L) +
                        β*Ev[a1]
                    if v > vstar
                        vstar = v
                        Lstar = L
                        a1star = a1
                    end
                end
                Vdict[i, s, t] = vstar
                Ldict[i, s, t] = Lstar
                A1dict[i, s, t] = a1star
                Cdict[i, s, t] = Ldict[i, s, t] * (w[t] + ξ[i]) + grid_A[s] * (1+r) - A1dict[i, s, t]
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

@everywhere function utility(c::R, L::R) where R <: Real
    if c <= 0 || 1 - L <= 0
        return -1e6
    else
        return log(c) + log(1 - L)
    end
end

@everywhere T = 65 # terminal period
@everywhere β = 0.95 # discount factor
@everywhere r = 0.05 # interest rate
@everywhere dist = LogNormal(0, 1) # distribution of ξ
@everywhere n = 5 # number of points of support of ξ

@everywhere w = Vector{Float64}(undef, T) # exogenous wages
w .= (900 .+ 20.0 .* (1:T) .- 0.5 .* (1:T).^2)

@everywhere ξ = quantile.(dist, 0:1/n:prevfloat(1.0))

@everywhere grid_A = -1_000:10.0:10_000

Vdict = SharedArray{Float64}((n, length(grid_A), T))
Cdict = similar(Vdict)
Ldict = similar(Vdict)
A1dict = similar(Vdict)

@everywhere dp = (u=utility, T=T, β=β, r=r, n=n, w=w, grid_A=grid_A, ξ=ξ)

@time solvemodel!(dp, Ldict, Cdict, A1dict, Vdict)