
using ArgParse
using DifferentialEquations
using DiffEqCallbacks
using LinearAlgebra
using Random
using SparseArrays
using StatsBase
using Distributions
using Plots


# Piecewise linear functions

# Piecewise linear interpolation at multiple positions (specialized for Float64)
function interp1d(x::Array{Float64,1}, x0::Array{Float64,1}, y0::Array{Float64,1})
    # x - position to evaluate function
    # x0 - positions of discontinuities in function
    # y0 - values of function on each region: y = y0[i] for x0[i] < x < x0[i+1]


    y = zeros(Float64, size(x))
    jj = 1
    for ii in 1:length(y)
        if x0[1]<=x[ii]<x0[end]
            jj = findnext(val->val>x[ii],x0,jj)-1
            y[ii] = y0[jj] + (y0[jj+1]-y0[jj])/(x0[jj+1]-x0[jj])*(x[ii]-x0[jj])
        end
    end
    return y
end

# Piecewise linear interpolation at a single position
function interp1d(x::Number, x0::Array, y0::Array)
    # x - position to evaluate function
    # x0 - positions of discontinuities in function slope
    # y0 - values of function at each discontinuity. Function takes the value 0 outside the range [x0[1], x0[length(x0)]]

    jj = 1
    y = 0.0
    if x0[1]<=x<x0[end]
        jj = findnext(val->val>x,x0,jj)-1
        y = y0[jj] + (y0[jj+1]-y0[jj])/(x0[jj+1]-x0[jj])*(x-x0[jj])
    end
    return y
end


# Piecewise constant interpolation
function piecewise(x::Number, x0::Array, y0::Array)
    # x - position to evaluate function
    # x0 - positions of discontinuities in function
    # y0 - values of function on each region: y = y0[i] for x0[i] < x < x0[i+1]

    jj = 1
    y = 0.0
    if x0[1]<=x<x0[end]
        jj = findnext(val->val>x,x0,jj)-1
        y = y0[jj]
    end
    return y
end


# Inverse of a piecewise constant density function
function piecewise_density_inv(x0::Array{T,1}, y0::Array{T,1}) where {T <: Number}
    csum = zeros(T, size(x0))
    for i in 2:length(x0)
        csum[i] = (x0[i] - x0[i-1])*y0[i-1] + csum[i-1]
    end
    csum = csum / csum[end]
    csum, x0
end


# Inverse of a piecewise linear density function
function linear_density_inverse(x0::Array{T,1}, y0::Array{T,1}) where {T <: Number}
    csum = zeros(T, size(x0))
    for	i in 2:length(x0)
    	csum[i] = (x0[i] - x0[i-1])*(y0[i-1] + y0[i])/2	+ csum[i-1]
    end
    csum_t = csum[end]
    function inverse(x::T)
        z = x*csum_t
        jj = findnext(val->val>z, csum, 1)-1
        z = z - csum[jj]
        y = y0[jj]
        dy = (y0[jj+1] - y0[jj])/(x0[jj+1]-x0[jj])
        if abs(dy)>1e-12
            return x0[jj]+((sqrt(y*y+2*z*dy)-y)/dy)
        elseif abs(y)>1e-12
            return x0[jj]+z/y
        else
            return x0[jj]
        end
    end
    inverse
end

# RI HEI10 escape rate function (\beta(C)*C in equation (2) )
function betaC(C, K, n_c)
    C ./ (1 .+ (C / K) .^ n_c)
end

# RI HEI10 escape rate function \beta(C)*C (evaluated in place)
function betaC!(dC, C, K, n_c)
    @. dC = C ./ (1 .+ (C ./ K) .^ n_c)
end

# Derivative of (\beta(C)*C) wrt C
function d_betaC(C::Float64, K::Float64, n_c::Float64)::Float64
    1 ./ (1.0 .+ (C ./ K) .^ n_c) .-
    C ./ (1.0 .+ (C ./ K) .^ n_c) .^ 2 .* (n_c ./ K) .* (C ./ K) .^ (n_c - 1)
end

# Derivative of (\beta(C)*C) wrt C (evaluated in place)
function d_betaC!(dbC, C, K, n_c)
    @. dbC = 1 ./ (1.0 .+ (C ./ K) .^ n_c) .-
    C ./ (1.0 .+ (C ./ K) .^ n_c) .^ 2 .* (n_c ./ K) .* (C ./ K) .^ (n_c - 1)
end

### Distributions of chromosome lengths

Lm = Vector{Float64}([37.5, 40.4, 45.7, 53.8, 59.8])
Ls = Vector{Float64}([4.93, 5.06, 5.86, 7.18, 7.13])


function basic_ode!(dyy::Array{Float64,1}, yy::Array{Float64,1}, p, t::Float64)
    N_array,           # Number of RIs (array)
    Q,           # Number of SCs
    K,           # Hill threshold
    a_nodes, # Rate of HEI10 absorption from pool to nodes
    betaC!,      # beta(C)*C function (passed to ODE for efficiency)
    d_betaC,     # derivative of beta(C)*C function wrt C
    gamma,         # Hill coefficient (\gamma)
    beta_2,    # RI HEI10 escape rate into pool (C-dependent)
    beta_3,    # RI HEI10 escape rate into pool (const) - This is zero in all the simulations in the paper
    bC  = p        # Placeholder to re-use vector

    @inbounds pool = yy[end]

    @inbounds dyy[end] = 0

    offset = 0
    for q = 1:Q

        @inbounds N = N_array[q]

        # Extract variables from ODE State vector

        @inbounds C = @view yy[offset+1:offset+N] # HEI10 amounts at the RH (C_j 1<=j<=N)
        @inbounds dC = @view dyy[offset+1:offset+N]

        # Calculate escape of HEI10 into nucleoplasm

        bbC = @view bC[1:N]
        betaC!(bbC, max.(C,0.0), K, gamma)

        # Rate of change of HEI10 amounts at the RIs
        @. dC = - beta_2 .* bC[1:N] .- beta_3 .* C + a_nodes*pool

        # Rate of change of HEI10 in nuceloplasmic pool

        @inbounds dyy[end] = dyy[end] -  a_nodes*pool*N + beta_2 *sum(bC[1:N]) + beta_3 *sum(C)
        offset += (N)
    end
end


function jac!(J, yy, p, t)
    N_array,           # Number of RIs (array)
    Q,           # Number of SCs
    K,           # Hill threshold
    a_nodes,     # From pool onto RI
    betaC!,      # beta(C)*C function (passed to ODE for efficiency)
    d_betaC,     # derivative of beta(C)*C function wrt C
    gamma,         # Hill coefficient (\gamma)
    beta_2,    # RI HEI10 escape rate into pool (C-dependent)
    beta_3,    # RI HEI10 escape rate into pool (const)
    bC = p        # Placeholder to re-use vector

    J[end, end] = 0
    offset = 0

    for q in 1:Q
        N = N_array[q]
        C = @view yy[offset+1:offset+N]

        dbC::Float64 = 0.0
        idx::Int64 = 0
        @inbounds for j = 1:N
            dbC = d_betaC(max(0, C[j]), K, gamma)
            J[offset+j, offset+j] = -(beta_2) * dbC - beta_3
            J[offset+j, end] = a_nodes
            J[end, offset+j] = beta_2*dbC + beta_3
        end
        J[end, end] = J[end, end] - N*a_nodes
        offset += N
    end
end

function simulate_video(idx, args)

    # Initialize random number generator with seed for this simulation

    L_array = Vector{Float64}()
    x_array = Vector{Vector{Float64}}()
    N_array = Vector{Int64}()

    y0 = Vector{Float64}()

    Q = size(Lm, 1)

    M_start = 0.0
    mt = MersenneTwister(idx)

    # Multiplicative noise in the overall level of HEI10 (unused)
    ratio_hei10 = exp(args["sd_hei10"]*randn(mt))

    use_poisson = Bool(args["use_poisson"])

    for q = 1:Q

        # Generate SC length, sampled from Normal distribution clipped to +/- 3 S.D.
        L::Float64 = Float64(Lm[q])+Float64(Ls[q])*clamp(randn(mt), -3, 3)
        push!(L_array, L)

	N::Int64 = 0
        # Calculate appropriate number of RIs for this SC length
	if use_poisson
	    N = rand(mt, Poisson(L*args["density"]))
	else
            N = floor(L*args["density"])
        end
        push!(N_array, N)



        x::Array{Float64,1} = L * sort(rand(mt, N))
        push!(x_array, x)

        # Initialize truncated normal distribution for RI HEI10 loading
        # Mean C0_noise (C_0)
        # Standard deviation C0_noise (\sigma)
        # Truncated at +/- 3 s.d.
        d = Truncated(Normal(Float64(args["C0"]), Float64(args["C0_noise"])), max(0, Float64(args["C0"])-3*Float64(args["C0_noise"])), Float64(args["C0"])+3*Float64(args["C0_noise"]))


        # Generate initial RI HEI10 amounts
        C0::Array{Float64,1} = rand(mt, d, N)
        C0[C0.<0] .= 0.0 # Double-check that initial RI HEI10 concentrations are non-negative

        C0_t = 1.0
        C0_t2::Float64 = Float64(args["t_C0_ratio2"]) # ($f_e$)
        t_L::Float64 = Float64(args["t_L"])


        for i in 1:N
    	    p = x[i]/L
            if p < t_L
               r = p/t_L
               C0[i] *= C0_t*(r) + C0_t2*(1-r)
            elseif p> 1-t_L
               r = (1-p)/t_L
               C0[i] *= C0_t*(r) + C0_t2*(1-r)
       	     end
	    end

        C0 *= ratio_hei10

        append!(y0, C0)
        M_start += sum(C0)
    end

    gamma::Float64 = Float64(args["gamma"]) # Hill coefficient (gamma)
    K::Float64 = Float64(args["K"]) # Hill threshold (K_C)

    a_nodes::Float64 = Float64(args["a_nodes"])
    beta_2 = Float64(args["beta_2"]) # RI escape rate (loss to nuceloplasm) - not used here
    beta_3 = Float64(args["beta_3"]) # RI constant escape rate (loss to nuceloplasm) - not used here
    bC = zeros(Float64, sum(N_array)) # Buffer to store values of \beta(C)*C in ODE calculation avoiding allocating memory

    pool0 = Float64(args["pool0"])*ratio_hei10

    M_start += pool0

    # Construct initial ODE state vector
    push!(y0, pool0)

    # ODE system parameters
    p = (
        N_array,
        Q,
        K,
        a_nodes,
        betaC!,
        d_betaC,
        gamma,
        beta_2,
        beta_3,
        bC,
    )

    # Initialize ODE problem with jacobian

    @show(y0)

    yy0 = zero(y0)
    basic_ode!(yy0, y0, p, 0.0)

    @show(yy0)

    # Initialize ODE problem with jacobian
    j = spzeros(sum(N_array) + 1, sum(N_array) + 1)
    g2 = ODEFunction(basic_ode!; jac=jac!, jac_prototype = j)
    prob_g = ODEProblem(g2, y0, (0.0, Float64(args["n_ts"]) * Float64(args["dt"])), p)
    cb = PositiveDomain(g2) # Additional constraint to ensure non-negativity of concentration values during integration

    # Solve ODE system using implicit Rodas5 solver, with tight relative error tolerance
    sol = solve(prob_g,Rodas5(autodiff=:false), cb=cb, save_everystep=true)#, reltol=1e-10, abstol=1e-10)

    offset = 0

    u_array = Vector{Array{Float64,2}}()
    u0_array = Vector{Vector{Float64}}()

    u_end = sol.u[end]
    t = range(0, stop=sol.t[end], length=100)
    u = hcat(sol(t)...)'
    for q in 1:Q
        z = u[:, offset+1:offset+N_array[q]]
        push!(u_array, z)

        z = y0[offset+1:offset+N_array[q]]
        push!(u0_array, z)

        offset += N_array[q]
    end

    (t, L_array, x_array, u_array, u, offset, u0_array)
end




function parse_commandline()

# Parse command line options and parameters

    s = ArgParseSettings()

    @add_arg_table! s begin
        "--filename"
            default="kymo.dat"
            help="base filename"

	"--timecourse"
        action = :store_true
	    help="timecourse instead"

	"--start"
           arg_type=Int
           default=1
           help="first seed to use for the simulation"
        "--n_ts"
            arg_type=Int
            default=100
            help="number of timesteps"

        "--dt"
     	    arg_type=Float64
	        default=360.0 # (s)
            help="time interval" # Length of each integration period (timestep is shorter, controlled automatically by error estimate)

        "--K"
     	    arg_type=Float64
            default=1.0 # (a.u.)
            help="threshold for RI unbinding" # (K_C) Hill function thresold for escape of HEI10 from RI

        "--gamma"
    	    arg_type=Float64
    	    default=1.25 # (no units)
            help="Hill coefficient for RI unbinding" # (\gamma) Hill coefficient for escape of HEI10 from RI

        "--pool0"
            arg_type=Float64
            default=280.0 # (a.u.)
            help="initial pool HEI10 concentration" # (c_0) HEI10 initial loading on SC

        "--C0"
            arg_type=Float64
            default=6.8 # (a.u.)
            help="initial RI HEI10" # (C_0) HEI10 initial RI loading

        "--C0_noise"
            arg_type=Float64
            default=2.2 # (a.u.)
            help="initial RI HEI10 noise" # (\sigma) Noise in HEI10 initial RI loading

        "--t_C0_ratio2"
            arg_type=Float64
            default=1.25 # (no units)
            help="initial hei10 ratio for RIs in telomeres"  # (f_e) Increased RI loading at SC telomeres

        "--t_L"
            arg_type=Float64
            default=0.2 # (no units)
            help="telomere rel length" # (x_e) Telomere length as fraction of bivalent

        "--density"
            arg_type=Float64
            default=0.5         # (um^{-1})
            help="RI density"  # (\rho) SC RI density

        "--a_nodes"
            arg_type=Float64   # (s^{-1})
            default=0.17       # Rate of HEI10 absorption from pool to RIs

         "--beta_2"
            arg_type=Float64   # (s^{-1})
            default=0.04        # RI HEI10 escape rate into pool (C-dependent)

        "--beta_3"
            arg_type=Float64   # (s^{-1})
            default=0.0        # RI HEI10 escape rate into pool (const) - This is zero in all the simulations in the paper

        "--Q"
            arg_type=Int64  
            default=5          # Number of chromosomes

        "--sd_hei10"
            arg_type=Float64   # (a.u.)
            default=0.0        # Variation in total HEI10 amounts

        "--use_poisson"
            arg_type=Bool      # Whether to distribute RIs at a constant rate per uniform length
            default=false      # (rather than a number proportional to chromosome length)
    end
    return parse_args(s)
end


function kymo_sol()
    n_args = parse_commandline()
    println("Parsed args:")

    for (arg,val) in n_args
        println("$arg=$val")
    end




    t, L_array, x_array, u_array, u, n, u0_array = simulate_video(n_args["start"], n_args)

#    plot((u[:, 1:n]), legend=false)


    open(n_args["filename"],"w") do io
       for i=1:5
           L = L_array[i]
           x = x_array[i]
           print(io, string(L) * ", " *string(t[end]) *'\n'*join(map(string, x), ',')*'\n')
           for j=1:100
               v=u_array[i][j,:]
               print(io, join(map(string, v), ',')*'\n')
           end
        end
    end

end

kymo_sol()
