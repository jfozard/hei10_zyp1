
using ArgParse
using DifferentialEquations
using DiffEqCallbacks
using LinearAlgebra
using Random
using SparseArrays
using StatsBase
#using NLopt
#using StatsPlots
using FiniteDiff
#using Plots
using Distributions
#using GLM
using Future

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
    @. dC =  C ./ (1 .+ (C ./ K) .^ n_c)
end

# Derivative of (\beta(C)*C) wrt C
function d_betaC(C::Float64, K::Float64, n_c::Float64)::Float64
    1 ./ (1.0 .+ (C ./ K) .^ n_c) .- C ./ (1.0 .+ (C ./ K) .^ n_c) .^ 2 .* (n_c ./ K) .* (C ./ K) .^ (n_c - 1)
end

# Derivative of (\beta(C)*C) wrt C (evaluated in place)
function d_betaC!(dbC, C, K, n_c)
    @. dbC = 1 ./ (1.0 .+ (C ./ K) .^ n_c) .- C ./ (1.0 .+ (C ./ K) .^ n_c) .^ 2 .* (n_c ./ K) .* (C ./ K) .^ (n_c - 1)
end




function Hs(t; s=10.0)
    0.5 .* (tanh.(t.*s) .+ 1)
end

function dHs(t; s=10.0)
    0.5 *s .* ( 1.0./ cosh.(t.*s).^2)
end

### Distributions of chromosome lengths

Lm = Vector{Float64}([37.5, 40.4, 45.7, 53.8, 59.8])
Ls = Vector{Float64}([4.93, 5.06, 5.86, 7.18, 7.13])


function basic_ode!(dyy::Array{Float64,1}, yy::Array{Float64,1}, p, t::Float64)
    m_array,           # Number of SC spatial compartments for discretization (array)
    N_array,           # Number of RIs (array)
    Q,           # Number of SCs
    K,           # Hill threshold
    K_u,         # Cutoff for recycling
    nodes_array,       # Indices of SC spatial compartments containing RIs
    D,           # RI Diffusion coefficient on SC (D_c)
    dL_array,          # Length of each spatial compartment (array)
    betaC,      # beta(C)*C function (passed to ODE for efficiency)
    d_betaC,     # derivative of beta(C)*C function wrt C
    gamma,         # Hill coefficient (\gamma)
    alpha,     # RI HEI10 absorption rate from SC (\alpha)
    beta,     # RI HEI10 escape rate back onto SC (\beta)
    bC,          # Placeholder to re-use vector
    A = p

    offset = 0
    for q = 1:Q

        @inbounds m = m_array[q]
        @inbounds N = N_array[q]
        @inbounds nodes = nodes_array[q]
        @inbounds dL = dL_array[q]

        # Extract variables from ODE State vector

    # Extract variables from ODE State vector
        @inbounds y = @view yy[offset+1:offset+m] # HEI10 concentration on the SC (c_i, 1<=i<=m)
        @inbounds dy = @view dyy[offset+1:offset+m]
        @inbounds C = @view yy[offset+m+1:offset+m+N] # HEI10 amounts at the RH (C_j 1<=j<=N)
        @inbounds dC = @view dyy[offset+m+1:offset+m+N]


        h = D/dL/dL

        @inbounds dy[1] = h * (y[2] - y[1])
        @inbounds @simd for i = 2:(m-1)
            dy[i] = h * (y[i+1] - 2 * y[i] + y[i-1])
        end

        @inbounds dy[m] = h * (y[m-1] - y[m])

        # Calculate escape of HEI10 back onto SC.

        # Calculate net fluxes from SC to RIs, and update ODE RHS for appropriate SC compartment
        @inbounds for j = 1:N
            bC_j = betaC(max.(C[j], 0.0), K, gamma) 
            A[j] = (alpha * y[nodes[j]] - beta * bC_j)*Hs(C[j] .- K_u)
            dyy[nodes[j]+offset] = dyy[nodes[j]+offset] - A[j] / dL
        end

        # Rate of change of HEI10 amounts at the RIs
        @. dC = @view A[1:N]

        offset += (m + N)
    end
end

function jac!(J, yy, p, t)
    m_array,           # Number of SC spatial compartments for discretization (array)
    N_array,           # Number of RIs (array)
    Q,           # Number of SCs
    K,           # Hill threshold
    K_u,         # Cutoff for recycling
    nodes_array,       # Indices of SC spatial compartments containing RIs
    D,           # RI Diffusion coefficient on SC (D_c)
    dL_array,          # Length of each spatial compartment (array)
    betaC,      # beta(C)*C function (passed to ODE for efficiency)
    d_betaC,     # derivative of beta(C)*C function wrt C
    gamma,         # Hill coefficient (\gamma)
    alpha,     # RI HEI10 absorption rate from SC (\alpha)
    beta,     # RI HEI10 escape rate back onto SC (\beta)
    bC,          # Placeholder to re-use vector
    A = p

    offset = 0

    for q in 1:Q
        nodes = nodes_array[q]
        m = m_array[q]
        N = N_array[q]
        dL = dL_array[q]
        r = D / dL / dL
        @inbounds y = @view yy[offset+1:offset+m] 
        C = @view yy[offset+m+1:offset+m+N]
        #bbC = @view bC[1:N]
        #betaC!(bbC, max.(C,0.0), K, gamma)
        J[offset+1, offset+1] = - r
        J[offset+1, offset+2] = r
        @inbounds for j = 2:m-1
            J[offset+j, offset+j-1] = r
            J[offset+j, offset+j] = - 2 * r
            J[offset+j, offset+j+1] = r
        end
        @inbounds J[offset+m, offset+m-1] = r
        @inbounds J[offset+m, offset+m] = - r
        dbC::Float64 = 0.0
        idx::Int64 = 0
        @inbounds for j = 1:N
            
            bbC_j = betaC(max(0, C[j]), K, gamma)
            dbC = d_betaC(max(0, C[j]), K, gamma)
	    Hs_j = Hs(C[j] .- K_u)
	    dHs_j = dHs(C[j] .- K_u)
	    y_j = y[nodes[j]]
	    J[offset+m+j, offset+m+j] = -beta * dbC *Hs_j + (alpha*y_j - beta * bbC_j) * dHs_j
            J[offset+nodes[j], offset+m+j] = (beta * dbC *Hs_j - (alpha*y_j - beta * bbC_j) * dHs_j)/dL
            J[offset+m+j, offset+nodes[j]] = alpha*Hs_j
            J[offset+nodes[j], offset+nodes[j]] -= alpha*Hs_j / dL 
        end
        offset += m+N
    end
end


function simulate_multi(idx, args)

    # Initialize random number generator with seed for this simulation

    L_array = Vector{Float64}()
    dL_array = Vector{Float64}()
    m_array = Vector{Int64}()
    nodes_array = Vector{Vector{Int64}}()
    x_array = Vector{Vector{Float64}}()
    N_array = Vector{Int64}()
    C0_array = Vector{Vector{Float64}}()

    y0 = Vector{Float64}()

    #Q = size(Lm, 1)

    C0_init = 0.0
    L_tot = 0.0
    mt = MersenneTwister(idx)

    # Multiplicative noise in the overall level of HEI10 (unused)

    use_poisson = Bool(args["use_poisson"])
    reset_seeds = Bool(args["reset_seeds"])

    Q = args["Q"]
    Q_s = args["Q_s"]

    for q = 1:Q
        if reset_seeds
    	    mt = MersenneTwister(idx)
        end
        # Generate SC length, sampled from Normal distribution clipped to +/- 3 S.D.
        L::Float64 = Float64(Lm[q+Q_s-1])+Float64(Ls[q+Q_s-1])*clamp(randn(mt), -3, 3)
        push!(L_array, L)

        L_tot += L        

	N::Int64 = 0
        # Calculate appropriate number of RIs for this SC length
	    if use_poisson
	        N = rand(mt, Poisson(L*args["density"]))
	    else
            N = round(L*args["density"])
        end
        push!(N_array, N)

        # Number of intervals for spatial discretization of SC
        m::Int64 = Int64(args["m"])
        push!(m_array, m)

        x::Array{Float64,1} = L * sort(rand(mt, N))
        push!(x_array, x)

        dL::Float64 = L / m
        push!(dL_array, dL)

    # Calculate indices of SC intervals containing each RI
        nodes = map(a -> ceil(Int, a), x / dL)
        push!(nodes_array, nodes)


    # Midpoints of each SC interval
        xm = (0.5:1.0:m) * dL

        #u0::Array{Float64,1} = fill(Float64(args["u0"]), m)

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

	push!(C0_array, C0)
        C0_init += sum(C0)
    end


    gamma::Float64 = Float64(args["gamma"]) # Hill coefficient (gamma)
    K::Float64 = Float64(args["K"]) # Hill threshold (K_C)
    K_u::Float64 = Float64(args["K_u"]) # Cutoff for recycling

    alpha::Float64 = Float64(args["alpha"]) # RI HEI10 absorption rate from SC (\alpha)
    beta::Float64 = Float64(args["beta"]) # RI HEI10 escape rate back onto SC (\beta)
    D::Float64 = Float64(args["D"]) # RI Diffusion coefficient on SC (D_c)

    bC = zeros(Float64, maximum(N_array)) # Buffer to store values of \beta(C)*C in ODE calculation avoiding allocating memory
    A = zeros(Float64, maximum(N_array)) # Buffer to store fluxes of HEI10 from RI to SC in ODE calculation

    # Construct initial ODE state vector

    M_tot_mean = Float64(args["M_tot_mean"])
    M_tot_sd = Float64(args["M_tot_sd"])




    d_M = Truncated(Normal(M_tot_mean, M_tot_sd), max(0, M_tot_mean-3*M_tot_sd), M_tot_mean+3*M_tot_sd)

    M_tot = rand(mt, d_M)


    C0_tot = M_tot * Float64(args["C0_ratio"])
    u0_tot = M_tot * Float64(args["u0_ratio"])
    u0 = u0_tot / L_tot


    M_start = 0.0
    
    for q = 1:Q
    	C0_array[q] = C0_array[q]*(C0_tot/C0_init)
	M_start += sum(C0_array[q])
    end

    for q = 1:Q
        m = m_array[q]
        u0_q::Array{Float64,1} = fill(Float64(u0), m)
        append!(y0, vcat(u0_q, C0_array[q]))
        M_start += u0*m*dL_array[q]
    end


       

    # ODE system parameters
    p = (
        m_array,
        N_array,
        Q,
        K,
	K_u,
        nodes_array,
        D,
        dL_array,
        betaC,
        d_betaC,
	gamma,
        alpha,
        beta,
        bC,
        A,
    )
    yy0 = zero(y0)
    basic_ode!(yy0, y0, p, 0.0)


    # Initialize ODE problem with jacobian
    j = spzeros(sum(m_array) + sum(N_array), sum(m_array) + sum(N_array))
    j2 = spzeros(sum(m_array) + sum(N_array), sum(m_array) + sum(N_array))
    
    y0_test = 1.0*rand(size(y0)...).+0.5
    jac!(j2, y0_test, p, 0.0)

    if idx==1

        @show(y0)
        j_fd = zeros(sum(m_array) + sum(N_array) , sum(m_array) + sum(N_array) )
        FiniteDiff.finite_difference_jacobian!(j_fd, (y,x)-> basic_ode!(y, x, p, 0.0), y0_test)
	 #   @show("check M0_start")
	 #   @show(M_start - M_tot)

        @show("check jac")
        @show(maximum(abs.(j_fd - j2)))
    end


    g2 = ODEFunction(basic_ode!; jac=jac!, jac_prototype = j2)
    prob_g = ODEProblem(g2, y0, (0.0, Float64(args["n_ts"]) * Float64(args["dt"])), p)
    cb = PositiveDomain()

    # Solve ODE system using implicit Rodas5 solver, with tight relative error tolerance
    sol = solve(prob_g,Rodas5(autodiff=:false), callback=cb, reltol=1e-12,  save_everystep=false)


    offset = 0
    u_array = Vector{Vector{Float64}}()
    u0_array = Vector{Vector{Float64}}()

    sol_end = sol.u[end]
    for q in 1:Q

        z = sol_end[offset+m_array[q]+1:offset+m_array[q]+N_array[q]]
        push!(u_array, z)

        z = y0[offset+m_array[q]+1:offset+m_array[q]+N_array[q]]
        push!(u0_array, z)

        offset += m_array[q] + N_array[q]
    end

    (L_array, x_array, u_array, u0_array)
end


function hist_hard(x::Vector{Float64}, n::Int64)
    h = zeros(Float64, n)
    for p in x
        h[min(max(1, Int(ceil(p * n))), n)] += 1
    end
    h
end


function get_counts(d, low, high)
    counts(d, low:high)
end

struct SimResult
    x_array::Vector{Vector{Float64}}
    u_array::Vector{Vector{Float64}}
    L_array::Vector{Float64}
    n_seed::Int
end

function simulate_all(args, io)

    start::Int64 = Int64(args["start"])
    n::Int64 = Int64(args["n"])
    @show(n)
    n_args = copy(args)

    solution_output = Vector{Vector{SimResult}}()
    for i in 1:Threads.nthreads()
        push!(solution_output, Vector{SimResult}())
    end

    Threads.@threads for i in start:start+(n-1)
        L_array, x_array, u_array, u0_array = simulate_multi(i, n_args)
        s = SimResult(x_array, u_array, L_array, i)
        push!(solution_output[Threads.threadid()], s)
    end

    solution_output = vcat(solution_output...)

    solution_seed = [s.n_seed for s in solution_output]
    solution_output = solution_output[sortperm(solution_seed)]

    for i in 1:length(solution_output)
        x_array = solution_output[i].x_array
        u_array = solution_output[i].u_array
        L_array = solution_output[i].L_array

        print(io, join(map(string, L_array), ',') * '\n')
        for q in 1:length(L_array)
            print(io, join(map(string, x_array[q]), ',') * '\n')
            print(io, join(map(string, u_array[q]), ',') * '\n')
        end
    end


end


function parse_commandline()

# Parse command line options and parameters

    s = ArgParseSettings()

    @add_arg_table! s begin
        "--filename_base"
            default="test"
            help="base filename"

        "--start"
            arg_type=Int
            default=1
            help="first seed to use for the simulation"

        "--n"
            arg_type=Int
            default=10 # (No unit)
            help="number of seeds" # Number of different SCs to simulate, using a different RNG seed for each one

        "--n_ts"
            arg_type=Int
            default=100
            help="number of timesteps"

	"--m"
            arg_type=Int
            default=2000
            help="number of segments for spatial discretization"
        "--dt"
     	    arg_type=Float64
            default=360.0 # (s)
            help="time interval" # Length of each integration period (timestep is shorter, controlled automatically by error estimate)
        "--K"
     	    arg_type=Float64
            default=1.0 # (a.u.)
            help="threshold for RI unbinding" # (K_C) Hill function thresold for escape of HEI10 from RI

        "--K_u"
     	    arg_type=Float64
            default=0.5 # (a.u.)
            help="threshold for RI HEI10 uptake (a.u.)"

        "--gamma"
    	    arg_type=Float64
    	    default=1.25 # (no units)
            help="Hill coefficient for RI unbinding" # (\gamma) Hill coefficient for escape of HEI10 from RI

        "--C0"
            arg_type=Float64
            default=1.0 # (a.u.)
            help="initial RI HEI10" # (C_0) HEI10 initial RI loading

        "--C0_noise"
            arg_type=Float64
            default=0.33 # (a.u.)
            help="initial RI HEI10 noise" # (\sigma) Noise in HEI10 initial RI loading

        "--t_C0_ratio2"
            arg_type=Float64
            default=2.0 # (no units)
            help="initial hei10 ratio for RIs in telomeres"  # (f_e) Increased RI loading at SC telomeres

        "--t_L"
            arg_type=Float64
            default=0.1 # (no units)
            help="telomere rel length" # (x_e) Telomere length as fraction of bivalent

        "--density"
            arg_type=Float64
            default=0.5         # (um^{-1})
            help="RI density"  # (\rho) SC RI density


        "--alpha"
            arg_type=Float64
            default=2.1 # (um s^{-1})
            help="binding rate at RIs" # (\alpha) RI HEI10 absorption rate from SC
        "--beta"
            arg_type=Float64
            default=0.5 # (s^{-1})
            help="unbinding rate at RIs" # (\beta) RI HEI10 escape rate back onto SC
        "--D"
            arg_type=Float64
            default=1.1 # (um^2 s^{-1})
            help="HEI10 diffusion constant on SC"  # (D_c) RI Diffusion coefficient on SC

        "--sd_hei10"
            arg_type=Float64   # (a.u.)
            default=0.0        # Variation in total HEI10 amounts

        "--use_poisson"
            arg_type=Bool      # Whether to distribute RIs at a constant rate per uniform length
            default=false      # (rather than a number proportional to chromosome length)

        "--Q"
            arg_type=Int
            default=5

        "--Q_s"
            arg_type=Int
            default=1

        "--reset_seeds"
           arg_type=Bool
           default=false

        "--M_tot_mean"
            arg_type=Float64
            default=1170.0 # 237.2*(1.2+0.5*6.8 + 2*0.5*1.0*0.1*0.5*6.8  )
            help="fixed total initial HEI10 amount"

	"--M_tot_sd"
            arg_type=Float64
            default=75.0 # From simulations of loading. 
            help="standard deviation of total initial HEI10 amount"

	"--C0_ratio"
	   arg_type=Float64
	   default = 0.76 # (0.5*6.8 + 2*0.5*1.0*0.1*0.5*6.8)/(1.2+0.5*6.8 + 2*0.5*1.0*0.1*0.5*6.8)

	"--u0_ratio"
	   arg_type=Float64
	   default = 0.24 # (1.2)/(1.2+0.5*6.8 + 2*0.5*1.0*0.1*0.5*6.8)	

    end
    return parse_args(s)
end


function plot_obj(filename, p; n=100, m=2000)
    args = parse_commandline()
    println("Parsed args:")

    # Update simulation parameters from function call parameters
    n_args = copy(args)
    n_args["M_tot_sd"]=p[1]*sqrt(p[2]^2*(n_args["M_tot_mean"]^2 + n_args["M_tot_sd"]^2) + n_args["M_tot_sd"]^2)
    n_args["M_tot_mean"] *= p[1]

    n_args["m"] = m
    n_args["n"] = n
    
    for (arg,val) in n_args
        println("$arg=$val")
    end
    println("filename ", filename)
    open(filename,"w") do io
        # Write parameter values as first line of output file
        println(io, "#Namespace(" * join(map(x-> string(x[1]) *"="*string(x[2]), collect(n_args)), ",") * ")")

        # Run (group of) simulations
        simulate_all(n_args, io)
    end
    println("done")
end


@show("done load ends")

total_SC_L = sum(Lm)
@show(total_SC_L)
p = "../output/simulation_output/"



@time(plot_obj(p * "test_5_0.6_0.0.dat", [0.6, 0.0]; n=1000))
@time(plot_obj(p * "test_5_1_0.0.dat", [1, 0.0]; n=1000))
@time(plot_obj(p * "test_5_4.5_0.0.dat", [4.5, 0.0]; n=1000))

@time(plot_obj(p * "test_5_0.6_0.2.dat", [0.6, 0.2]; n=1000))
@time(plot_obj(p * "test_5_1_0.2.dat", [1, 0.2]; n=1000))
@time(plot_obj(p * "test_5_4.5_0.2.dat", [4.5, 0.2]; n=1000))

