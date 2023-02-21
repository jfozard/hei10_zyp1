
using ArgParse
using DifferentialEquations
using DiffEqCallbacks
using LinearAlgebra
using Random
using SparseArrays
using StatsBase
using FiniteDiff
using Distributions


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
    N_array,           # Number of RIs (array)
    Q,           # Number of SCs
    K,           # Hill threshold
    K_u,         # Cutoff for nucleoplasmic recycling
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
        @. dC = - beta_2 .* bC[1:N] .*Hs(C .- K_u) .- beta_3.*Hs(C .- K_u) .* C + a_nodes*pool.*Hs(C.-K_u)

        # Rate of change of HEI10 in nuceloplasmic pool

        @inbounds dyy[end] = dyy[end] -  a_nodes*pool*sum(Hs(C.-K_u)) + beta_2 *sum(bC[1:N] .* Hs(C.-K_u)) + beta_3 *sum(C .* Hs(C.-K_u))
        offset += (N)
    end
end


function jac!(J, yy, p, t)
    N_array,           # Number of RIs (array)
    Q,           # Number of SCs
    K,           # Hill threshold
    K_u,           # Hill threshold	
    a_nodes,     # From pool onto RI
    betaC!,      # beta(C)*C function (passed to ODE for efficiency)
    d_betaC,     # derivative of beta(C)*C function wrt C
    gamma,         # Hill coefficient (\gamma)
    beta_2,    # RI HEI10 escape rate into pool (C-dependent)
    beta_3,    # RI HEI10 escape rate into pool (const)
    bC = p        # Placeholder to re-use vector

    J[end, end] = 0
    offset = 0
    @inbounds pool = yy[end]

    for q in 1:Q
        N = N_array[q]
        C = @view yy[offset+1:offset+N]
        bbC = @view bC[1:N]
        betaC!(bbC, max.(C,0.0), K, gamma)
        dbC::Float64 = 0.0
        idx::Int64 = 0
        @inbounds for j = 1:N
            dbC = d_betaC(max(0, C[j]), K, gamma)
            J[offset+j, offset+j] = a_nodes*pool*dHs(C[j]-K_u)- beta_2*Hs(C[j]-K_u) * dbC -beta_2*dHs(C[j]-K_u )*bbC[j] - beta_3*Hs(C[j]-K_u) -beta_3*dHs(C[j]-K_u)*C[j]
            J[offset+j, end] = a_nodes*Hs(C[j]-K_u)
            J[end, offset+j] = beta_2*dbC*Hs(C[j]-K_u) + beta_2*bbC[j]*dHs(C[j]-K_u) + beta_3*Hs(C[j]-K_u) + beta_3*C[j]*dHs(C[j]-K_u) -a_nodes*pool*dHs(C[j]-K_u)
        end
        J[end, end] = J[end, end] - sum(Hs(C.-K_u))*a_nodes
        offset += N
    end
end


function simulate_multi(idx, args, Lm, Ls; tc=false)

    # Initialize random number generator with seed for this simulation

    L_array = Vector{Float64}()
    x_array = Vector{Vector{Float64}}()
    N_array = Vector{Int64}()
    C0_array = Vector{Vector{Float64}}()

    y0 = Vector{Float64}()

    Q = size(Lm, 1)

    C0_init = 0.0
    
    M_start = 0.0
    mt = MersenneTwister(idx)

    # Multiplicative noise in the overall level of HEI10 (unused)
    #ratio_hei10 = exp(args["sd_hei10"]*randn(mt))

    use_poisson = Bool(args["use_poisson"])

    for q = 1:Q
        #mt = MersenneTwister(idx)

        # Generate SC length, sampled from Normal distribution clipped to +/- 3 S.D.
        L::Float64 = Float64(Lm[q])+Float64(Ls[q])*clamp(randn(mt), -3, 3)
        push!(L_array, L)

	N::Int64 = 0
        # Calculate appropriate number of RIs for this SC length
	if use_poisson
	    N = rand(mt, Poisson(L*args["density"]))
	else
            N = round(L*args["density"])
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

	    push!(C0_array, C0)
        C0_init += sum(C0)

        #append!(y0, C0)
    end

    gamma::Float64 = Float64(args["gamma"]) # Hill coefficient (gamma)
    K::Float64 = Float64(args["K"]) # Hill threshold (K_C)
    K_u::Float64 = Float64(args["K_u"]) # Hill threshold (K_C)	

    a_nodes::Float64 = Float64(args["a_nodes"])
    beta_2 = Float64(args["beta_2"]) # RI escape rate (loss to nuceloplasm) - not used here
    beta_3 = Float64(args["beta_3"]) # RI constant escape rate (loss to nuceloplasm) - not used here
    bC = zeros(Float64, sum(N_array)) # Buffer to store values of \beta(C)*C in ODE calculation avoiding allocating memory

    # Construct initial ODE state vector

    M_tot_mean = Float64(args["M_tot_mean"])
    M_tot_sd = Float64(args["M_tot_sd"])

    d_M = Truncated(Normal(M_tot_mean, M_tot_sd), max(0, M_tot_mean-3*M_tot_sd), M_tot_mean+3*M_tot_sd)

    M_tot = rand(mt, d_M)

    C0_tot = M_tot * Float64(args["C0_ratio"])
    pool0 = M_tot * Float64(args["pool0_ratio"])    



    M_start = 0.0
    
    for q = 1:Q
    	C0_array[q] = C0_array[q]*(C0_tot/C0_init)
	    M_start += sum(C0_array[q])
        append!(y0, C0_array[q])
    end
    push!(y0, pool0)

    y0 = y0
    M_start = M_start + pool0
    

    # ODE system parameters
    p = (
        N_array,
        Q,
        K,
	    K_u,
        a_nodes,
        betaC!,
        d_betaC,
        gamma,
        beta_2,
        beta_3,
        bC,
    )

    # Initialize ODE problem with jacobian

    j2 = spzeros(sum(N_array) + 1, sum(N_array) + 1)

    yy0 = zero(y0)

    y0_test = 1.0*rand(size(y0)...).+0.5

    basic_ode!(yy0, y0, p, 0.0)
    jac!(j2, y0_test, p, 0.0)

    if idx==1

        j_fd = zeros(sum(N_array) + 1, sum(N_array) + 1)
        FiniteDiff.finite_difference_jacobian!(j_fd, (y,x)-> basic_ode!(y, x, p, 0.0), y0_test)

        @show("check jac")

        @show(maximum(abs.(j_fd - j2)./max.(abs.(j2), 1e-6)))

    end 

    # Initialize ODE problem with jacobian
    j = spzeros(sum(N_array) + 1, sum(N_array) + 1)
    g2 = ODEFunction(basic_ode!; jac=jac!, jac_prototype = j)
    prob_g = ODEProblem(g2, y0, (0.0, Float64(args["n_ts"]) * Float64(args["dt"])), p)
    cb = PositiveDomain(save=false)#g2) # Additional constraint to ensure non-negativity of concentration values during integration

    if !tc

    # Solve ODE system using implicit Rodas5 solver, with tight relative error tolerance
    sol = solve(prob_g,Rodas5(autodiff=:false), callback=cb, reltol=1e-12, save_everystep=true)

    offset = 0
    u_array = Vector{Vector{Float64}}()
    u0_array = Vector{Vector{Float64}}()

    u_end = sol.u[end]
    for q in 1:Q
        z = u_end[offset+1:offset+N_array[q]]
        push!(u_array, z)

        z = y0[offset+1:offset+N_array[q]]
        push!(u0_array, z)

        offset += N_array[q]
    end
    if idx<100
        @show(length(sol))
    end
    res=(L_array, x_array, u_array, u0_array)
    else
        sol = solve(prob_g,Rodas5(autodiff=:false), callback=cb, reltol=1e-12, save_everystep=true)

            t = sol.t
            offset = 0
            u_t_array = Vector{Array{Float64}}()
            u0_array = Vector{Vector{Float64}}()

            sol_end = sol.u[end]
            sol_u = transpose(hcat(sol.u...))
            for q in 1:Q

                z = sol_u[:, offset+1:offset+N_array[q]]
                push!(u_t_array, z)

                z = y0[offset+1:offset+N_array[q]]
                push!(u0_array, z)

                offset += N_array[q]
            end

            p_t_array = sol_u[:, end]
            res = (t, u_t_array, p_t_array, x_array, L_array)
    end
    res
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

struct SimResult_TC
    t ::Vector{Float64}
    u_t_array::Vector{Array{Float64, 2}}
    p_t::Vector{Float64}
    x_array::Vector{Vector{Float64}}
    L_array::Vector{Float64}
    n_seed::Int
end

function simulate_all(args, io)



    Q = Int64(args["Q"])

    start::Int64 = Int64(args["start"])
    n::Int64 = Int64(args["n"])
    n_args = copy(args)

    solution_output = Vector{Vector{SimResult}}()
    for i in 1:Threads.nthreads()
        push!(solution_output, Vector{SimResult}())
    end

    Threads.@threads for i in start:start+(n-1)
        L_array, x_array, u_array, u0_array = simulate_multi(i, n_args, Lm[1:Q], Ls[1:Q]; tc=false)
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

function simulate_all_tc(args, io)



    Q = Int64(args["Q"])


    start::Int64 = Int64(args["start"])
    n::Int64 = Int64(args["n"])
    @show(n)
    n_args = copy(args)

    solution_output = Vector{Vector{SimResult_TC}}()
    for i in 1:Threads.nthreads()
        push!(solution_output, Vector{SimResult_TC}())
    end

    Threads.@threads for i in start:start+(n-1)
        t, u_t_array, p_t_array, x_array, L_array = simulate_multi(i, n_args, Lm[1:Q], Ls[1:Q]; tc=true)
        s = SimResult_TC(t, u_t_array, p_t_array, x_array, L_array, i)
        push!(solution_output[Threads.threadid()], s)
    end

    solution_output = vcat(solution_output...)
    solution_seed = [s.n_seed for s in solution_output]
    solution_output = solution_output[sortperm(solution_seed)]

    for i in 1:length(solution_output)
        x_array = solution_output[i].x_array
        u_t_array = solution_output[i].u_t_array
        t = solution_output[i].t
        L_array = solution_output[i].L_array
        p_t = solution_output[i].p_t


        print(io, join(map(string, L_array), ',') * '\n')
        print(io, join(map(string, t), ',') * '\n')
        for q in 1:Q
            print(io, join(map(string, x_array[q]), ',') * '\n')
            for j in 1:length(t)
                print(io, join(map(string, u_t_array[q][j,:]), ',') * '\n')
            end
        end
        print(io, join(map(string, p_t), ',') * '\n')
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
            default=1000 # (No unit)
            help="number of seeds" # Number of different SCs to simulate, using a different RNG seed for each one

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
            default=1.25 # (no units)
            help="initial hei10 ratio for RIs in telomeres"  # (f_e) Increased RI loading at SC telomeres

        "--t_L"
            arg_type=Float64
            default=0.3 # (no units)
            help="telomere rel length" # (x_e) Telomere length as fraction of bivalent

        "--density"
            arg_type=Float64
            default=0.5         # (um^{-1})
            help="RI density"  # (\rho) SC RI density

        "--a_nodes"
            arg_type=Float64   # (s^{-1})
            default=0.14       # Rate of HEI10 absorption from pool to RIs


         "--beta_2"
            arg_type=Float64   # (s^{-1})
            default=0.034        # RI HEI10 escape rate into pool (C-dependent)

        "--beta_3"
            arg_type=Float64   # (s^{-1})
            default=0.0        # RI HEI10 escape rate into pool (const) - This is zero in all the simulations in the paper

        "--Q"
            arg_type=Int64
            default=5          # Number of chromosomes

        "--use_poisson"
            arg_type=Bool      # Whether to distribute RIs at a constant rate per uniform length
            default=true      # (rather than a number proportional to chromosome length)
    
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

	"--pool0_ratio"
	        arg_type=Float64
	        default = 0.24 # (1.2)/(1.2+0.5*6.8 + 2*0.5*1.0*0.1*0.5*6.8)	

    end
    return parse_args(s)
end


function plot_obj(filename, p; n=100, tc=false, regtot=true)
    args = parse_commandline()
    println("Parsed args:")

    # Update simulation parameters from function call parameters
    n_args = copy(args)
    n_args["n"] = n


    n_args["M_tot_sd"]=p[1]*sqrt(p[2]^2*(n_args["M_tot_mean"]^2 + n_args["M_tot_sd"]^2) + n_args["M_tot_sd"]^2)
    n_args["M_tot_mean"] *= p[1]   

    n_args["t_C0_ratio2"] = p[3]
    n_args["t_L"] = p[4]
    n_args["n_c"] = p[5]
    n_args["a_nodes"] = p[6]
    n_args["beta_2"] = p[7]

    for (arg,val) in n_args
        println("$arg=$val")
    end
    println("filename ", filename)
    open(filename,"w") do io
        # Write parameter values as first line of output file
        println(io, "#Namespace(" * join(map(x-> string(x[1]) *"="*string(x[2]), collect(n_args)), ",") * ")")

        # Run (group of) simulations
        if tc
            simulate_all_tc(n_args, io)
        else
            simulate_all(n_args, io)
        end
    end
    println("done")
end



@show("done load ends")

total_SC_L = sum(Lm)

@show(total_SC_L)


p = "../output/simulation_output/"

fac=3.5; @time plot_obj(p * "regtot-nuc-ox-poisson-0.0.dat",[fac, 0.0, 1.25, 0.3, 1.25, 0.14, 0.03]; n=10000)
fac=3.5; plot_obj(p * "regtot-nuc-ox-poisson-0.2.dat",[fac, 0.2, 1.25, 0.3, 1.25, 0.14, 0.03]; n=10000)
fac=1.0; plot_obj(p * "regtot-nuc-poisson-0.0.dat",[fac, 0.0, 1.25, 0.3, 1.25, 0.14, 0.03]; n=10000)
fac=1.0; plot_obj(p * "regtot-nuc-poisson-0.2.dat",[fac, 0.2, 1.25, 0.3, 1.25, 0.14, 0.03]; n=10000)
fac=1.0; plot_obj(p * "regtot-nuc-no-ends-poisson-0.0.dat",[fac, 0.0, 1.0, 0.3, 1.25, 0.14, 0.03]; n=10000)


fac=3.5; plot_obj(p * "regtot-nuc-ox-poisson-0.0.kymo",[fac, 0.0, 1.25, 0.3, 1.25, 0.14, 0.03]; n=1, tc=true)
fac=3.5; plot_obj(p * "regtot-nuc-ox-poisson-0.2.kymo",[fac, 0.2, 1.25, 0.3, 1.25, 0.14, 0.03]; n=1, tc=true)
fac=1.0; plot_obj(p * "regtot-nuc-poisson-0.0.kymo",[fac, 0.0, 1.25, 0.3, 1.25, 0.14, 0.03]; n=1, tc=true)
fac=1.0; plot_obj(p * "regtot-nuc-poisson-0.2.kymo",[fac, 0.2, 1.25, 0.3, 1.25, 0.14, 0.03]; n=1, tc=true)
