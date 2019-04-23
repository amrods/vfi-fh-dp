
using Parameters
using QuantEcon: rouwenhorst
using LinearAlgebra
using Interpolations
using Optim

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

    for i in 1:n
        for s in 1:length(grid_A)
            # use bisection here
            opt = optimize(x -> -utility(x*(w[T] + ξ[i]) + grid_A[s]*(1+r), x), 0.0, 1.0)
            xstar = Optim.minimizer(opt)
            Ldict[s, i, T] = xstar
            A1dict[s, i, T] = 0.0
            Cdict[s, i, T] = (w[T] + ξ[i])*Ldict[s, i, T] + grid_A[s]*(1+r)
            Vdict[s, i, T] = -Optim.minimum(opt)
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
        EV = LinearInterpolation( (grid_A, ξ), transpose( ℙ * transpose(Vdict[:, :, t+1])), extrapolation_bc = Line() )
        @time for i in 1:n
            for s in 1:length(grid_A)
                # x[1] is assets to carry forward, x[2] is labor supply
                initial_x = [A1dict[s, i, t+1], 0.0]
                opt = optimize(x -> -( utility(x[2]*(w[T] + ξ[i]) + grid_A[s]*(1+r) - x[1], x[2]) + β*EV(x[1], ξ[i]) ),
                        initial_x,
                        NelderMead(), Optim.Options(show_trace=false))
                xstar = Optim.minimizer(opt)
                Ldict[s, i, T] = xstar[2]
                A1dict[s, i, T] = xstar[1]
                Cdict[s, i, T] = (w[T] + ξ[i])*Ldict[s, i, T] + grid_A[s]*(1+r)
                Vdict[s, i, T] = -Optim.minimum(opt)
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
Model = @with_kw (u=utility, n=5, w, r=0.05, T=65, β=0.95, grid_A=-1_000:10.0:10_000, ρ=0.7, σ=15.0, μ=0.0)

dp = Model(w=w)

Vdict = Array{Float64}(undef, (length(dp.grid_A), dp.n, dp.T))
Cdict = Array{Float64}(undef, (length(dp.grid_A), dp.n, dp.T))
Ldict = Array{Float64}(undef, (length(dp.grid_A), dp.n, dp.T))
A1dict = Array{Float64}(undef, (length(dp.grid_A), dp.n, dp.T))

#@time solvemodel!(dp, Ldict, Cdict, A1dict, Vdict);
