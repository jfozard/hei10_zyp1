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

### Distributions of chromosome lengths

Lm = Vector{Float64}([37.5, 40.4, 45.7, 53.8, 59.8])
Ls = Vector{Float64}([4.93, 5.06, 5.86, 7.18, 7.13])

function simulate_multi(idx, args)

    # Initialize random number generator with seed for this simulation

    L_array = Vector{Float64}()
    dL_array = Vector{Float64}()
    m_array = Vector{Int64}()
    nodes_array = Vector{Vector{Int64}}()
    x_array = Vector{Vector{Float64}}()
    N_array = Vector{Int64}()

    y0 = Vector{Float64}()

    #Q = size(Lm, 1)

    M_start = 0.0
    M_C = 0.0
    mt = MersenneTwister(idx)

    # Multiplicative noise in the overall level of HEI10 (unused)


    sd_hei10 = Float64(args["sd_hei10"])
    ratio_hei10::Float64 = 1.0
    if sd_hei10>0
        d_ratio_hei10 = Truncated(Normal(1.0, sd_hei10), max(0, 1-3*sd_hei10), 1+3*sd_hei10)
        ratio_hei10 = rand(mt, d_ratio_hei10)
    end
    #ratio_hei10 = exp(args["sd_hei10"]*randn(mt))

    use_poisson = Bool(args["use_poisson"])
    reset_seeds = Bool(args["reset_seeds"])

    Q = args["Q"]
    Q_s = args["Q_s"]

    for q = 1:Q
        #if reset_seeds
    	#    mt = MersenneTwister(idx)
	#end
	#else
        #    mt = Future.randjump(mt, big(10)^20)
        #end

        # Generate SC length, sampled from Normal distribution clipped to +/- 3 S.D.
        L::Float64 = Float64(Lm[q+Q_s-1])+Float64(Ls[q+Q_s-1])*clamp(randn(mt), -3, 3)
        push!(L_array, L)

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

        u0::Array{Float64,1} = fill(Float64(args["u0"]), m)

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

        append!(y0, vcat(u0, C0))
        M_start += dL*sum(u0)+sum(C0)
	M_C += sum(C0)
    end
    M_start, M_C
end

function simulate_all(args, io)


    start::Int64 = Int64(args["start"])
    n::Int64 = Int64(args["n"])
    @show(n)
    n_args = copy(args)

    solution_output = Vector{Vector{Float64}}()
    for i in 1:Threads.nthreads()
        push!(solution_output, Vector{Float64}())
    end

    solution_output_C = Vector{Vector{Float64}}()
    for i in 1:Threads.nthreads()
        push!(solution_output_C, Vector{Float64}())
    end

    Threads.@threads for i in start:start+(n-1)
        s, c = simulate_multi(i, n_args)
        push!(solution_output[Threads.threadid()], s)
        push!(solution_output_C[Threads.threadid()], c)
    end

    solution_output = vcat(solution_output...)
    solution_output_C = vcat(solution_output_C...)

    for i in 1:length(solution_output)
        M = solution_output[i]
	C = solution_output_C[i]
        print(io, string(M) * ", " * string(C) * '\n')
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
        "--gamma"
    	    arg_type=Float64
    	    default=1.25 # (no units)
            help="Hill coefficient for RI unbinding" # (\gamma) Hill coefficient for escape of HEI10 from RI
        "--u0"
            arg_type=Float64
            default=1.2
            help="Initial HEI10 amount on SC"
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

    end
    return parse_args(s)
end


function plot_obj(filename, p; n=100, m=2000)
    args = parse_commandline()
    println("Parsed args:")

    # Update simulation parameters from function call parameters
    n_args = copy(args)
    n_args["C0"] *= p[1]
    n_args["C0_noise"]*=p[1]
    n_args["u0"]*=p[1]
    n_args["sd_hei10"]=p[2]
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

function plot_obj(filename, p; n=100, m=2000)
    args = parse_commandline()
    println("Parsed args:")

    # Update simulation parameters from function call parameters
    n_args = copy(args)
    n_args["C0"] *= p[1]
    n_args["C0_noise"]*=p[1]
    n_args["u0"]*=p[1]
    n_args["sd_hei10"]=p[2]

    n_args["m"] = m

    n_args["n"] = n
    for (arg,val) in n_args
        println("$arg=$val")
    end
    println("filename ", filename)
    open(filename,"w") do io
        # Write parameter values as first line of output file
#        println(io, "#Namespace(" * join(map(x-> string(x[1]) *"="*string(x[2]), collect(n_args)), ",") * ")")

        # Run (group of) simulations
        simulate_all(n_args, io)
    end
    println("done")
end

function plot_obj_new(filename, p; n=100, m=500)
    args = parse_commandline()
    println("Parsed args:")

    # Update simulation parameters from function call parameters
    n_args = copy(args)
    n_args["C0"] *= p[1]
    n_args["C0_noise"]*=p[1]
    n_args["u0"]*=p[1]
    n_args["sd_hei10"]=p[2]
    n_args["alpha"] = p[3]
    n_args["beta"] = p[4]
    n_args["density"] *= p[5]
    n_args["m"] = m
    n_args["t_C0_ratio2"] = p[6] #1.5
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
function plot_obj_new_mid(filename, p; n=100, m=500, n_ts=50)
    args = parse_commandline()
    println("Parsed args:")

    # Update simulation parameters from function call parameters
    n_args = copy(args)
    n_args["C0"] *= p[1]
    n_args["C0_noise"]*=p[1]
    n_args["u0"]*=p[1]
    n_args["sd_hei10"]=p[2]
    n_args["alpha"] = p[3]
    n_args["beta"] = p[4]
    n_args["density"] *= p[5]
    n_args["m"] = m
    n_args["t_C0_ratio2"] = 1.5
    n_args["n"] = n
    n_args["n_ts"] = n_ts
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
function plot_obj_first(filename, p; n=100, m=2000)
    args = parse_commandline()
    println("Parsed args:")

    # Update simulation parameters from function call parameters
    n_args = copy(args)
    n_args["C0"] *= p[1]
    n_args["C0_noise"]*=p[1]
    n_args["u0"]*=p[1]
    n_args["sd_hei10"]=p[2]
    n_args["m"] = m

    n_args["n"] = n
    n_args["Q"] = 1
    n_args["Q_s"] = 1

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

function plot_obj_last(filename, p; n=100, m=2000)
    args = parse_commandline()
    println("Parsed args:")

    # Update simulation parameters from function call parameters
    n_args = copy(args)
    n_args["C0"] *= p[1]
    n_args["C0_noise"]*=p[1]
    n_args["u0"]*=p[1]
    n_args["sd_hei10"]=p[2]
    n_args["m"] = m

    n_args["n"] = n
    n_args["Q"] = 1
    n_args["Q_s"] = 5

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


#@time(plot_obj("M_5_0.6_0.0.dat", [0.6, 0.0]; n=1000))
@time(plot_obj("M_5_1_0.0.dat", [1, 0.0]; n=100000))
#@time(plot_obj("M_5_4.5_0.0.dat", [4.5, 0.0]; n=1000))
