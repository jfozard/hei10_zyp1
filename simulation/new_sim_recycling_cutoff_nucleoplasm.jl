
include("common_recycling_both.jl")

function simulate_multi(idx, args, Lm, Ls; tc=false)

    # Initialize random number generator with seed for this simulation

    L_array = Vector{Float64}()
    dL_array = Vector{Float64}()
    m_array = Vector{Int64}()
    nodes_array = Vector{Vector{Int64}}()
    x_array = Vector{Vector{Float64}}()
    N_array = Vector{Int64}()
    C0_array = Vector{Vector{Float64}}()

    y0 = Vector{Float64}()

    Q = size(Lm, 1)

    C0_init = 0.0
    L_tot = 0.0
    mt = MersenneTwister(idx)


    use_poisson = Bool(args["use_poisson"])

    for q = 1:Q
        #mt = MersenneTwister(idx)

        # Generate SC length, sampled from Normal distribution clipped to +/- 3 S.D.
        L::Float64 = Float64(Lm[q])+Float64(Ls[q])*clamp(randn(mt), -3, 3)
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


    M_tot_mean = Float64(args["M_tot_mean"])
    M_tot_sd = Float64(args["M_tot_sd"])

    d_M = Truncated(Normal(M_tot_mean, M_tot_sd), max(0, M_tot_mean-3*M_tot_sd), M_tot_mean+3*M_tot_sd)



    d_M = Truncated(Normal(M_tot_mean, M_tot_sd), max(0, M_tot_mean-3*M_tot_sd), M_tot_mean+3*M_tot_sd)

    M_tot = rand(mt, d_M)


    C0_tot = M_tot * Float64(args["C0_ratio"])
    u0_tot = M_tot * Float64(args["u0_ratio"])
    u0 = u0_tot / L_tot


    M_start = 0.0
    
    for q = 1:Q
    	C0_array[q] = (C0_array[q]/C0_init)*C0_tot
	M_start += sum(C0_array[q])
    end

    for q = 1:Q
        m = m_array[q]
        u0_q::Array{Float64,1} = fill(Float64(u0), m)
        append!(y0, vcat(u0_q, C0_array[q]))
        M_start += u0*m*dL_array[q]
    end

    pool0 = 0.0
    push!(y0, pool0)

    gamma::Float64 = Float64(args["gamma"]) # Hill coefficient (gamma)
    K::Float64 = Float64(args["K"]) # Hill threshold (K_C)
    K_u::Float64 = Float64(args["K_u"]) # RI HEI10 uptake threshold (K_u)
    a_SC = Float64(args["a_SC"]) # HEI10 production rate on SC (Not used here)
    b_SC = Float64(args["b_SC"]) # HEI10 decay rate on SC (Not used here)
    alpha::Float64 = Float64(args["alpha"]) # RI HEI10 absorption rate from SC (\alpha)
    beta::Float64 = Float64(args["beta"]) # RI HEI10 escape rate back onto SC (\beta)
    a_nodes::Float64 = Float64(args["a_nodes"])
    beta_2 = Float64(args["beta_2"]) # RI escape rate (loss to nuceloplasm) - not used here
    beta_3 = Float64(args["beta_3"]) # RI constant escape rate (loss to nuceloplasm) - not used here
    D::Float64 = Float64(args["D"]) # RI Diffusion coefficient on SC (D_c)
    pool_decay::Float64 = Float64(args["pool_decay"])


    bC = zeros(Float64, maximum(N_array)) # Buffer to store values of \beta(C)*C in ODE calculation avoiding allocating memory
    A = zeros(Float64, maximum(N_array)) # Buffer to store fluxes of HEI10 from RI to SC in ODE calculation




    # ODE system parameters
    p = (
        m_array,
        N_array,
        Q,
        K,
        K_u,
        nodes_array,
        a_SC,
        b_SC,
        a_nodes,
        D,
        dL_array,
        betaC!,
        d_betaC,
	gamma,
        alpha,
        beta,
        beta_2,
        beta_3,
        pool_decay,
        bC,
        A,
    )
    yy0 = zero(y0)
    basic_ode_smooth!(yy0, y0, p, 0.0)


    # Initialize ODE problem with jacobian
    j = spzeros(sum(m_array) + sum(N_array) + 1, sum(m_array) + sum(N_array) + 1)
    j2 = spzeros(sum(m_array) + sum(N_array) + 1, sum(m_array) + sum(N_array) + 1)

    y0_test = 1.0*rand(size(y0)...).+0.5

    jac_smooth!(j2, y0_test, p, 0.0)

    if idx==1

        j_fd = zeros(sum(m_array) + sum(N_array) + 1, sum(m_array) + sum(N_array) + 1)
        FiniteDiff.finite_difference_jacobian!(j_fd, (y,x)-> basic_ode_smooth!(y, x, p, 0.0), y0_test)


        @show("check jac")
        @show(maximum(abs.(j_fd - j2)./max.(abs.(j2), 1e-6)))
    end

    g2 = ODEFunction(basic_ode_smooth!; jac=jac_smooth!, jac_prototype = j2)
    cb = PositiveDomain()#g2)
    prob_g = ODEProblem(g2, y0, (0.0, Float64(args["n_ts"]) * Float64(args["dt"])), p)


    if !tc
    # Solve ODE system using implicit Rodas5 solver, with tight relative error tolerance
    sol = solve(prob_g,Rodas5(autodiff=:false), callback=cb, reltol=1e-12, save_everystep=false)

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
    if idx==1
        @show(sol_end[end])
    end
    res = (L_array, x_array, u_array, u0_array)
    else
        # Solve ODE system using implicit Rodas5 solver, with tight relative error tolerance
    sol = solve(prob_g,Rodas5(autodiff=:false), callback=cb, retol=1e-12, save_everystep=true)

    t = sol.t
    offset = 0
    u_t_array = Vector{Array{Float64}}()
    c_t_array = Vector{Array{Float64}}()
    u0_array = Vector{Vector{Float64}}()

    sol_end = sol.u[end]
    sol_u = transpose(hcat(sol.u...))
    for q in 1:Q

        z = sol_u[:, offset+1:offset+m_array[q]]
        push!(c_t_array, z)

        z = sol_u[:, offset+m_array[q]+1:offset+m_array[q]+N_array[q]]
        push!(u_t_array, z)

        z = y0[offset+m_array[q]+1:offset+m_array[q]+N_array[q]]
        push!(u0_array, z)

        offset += m_array[q] + N_array[q]
    end

    p_t_array = sol_u[:, end]
    res = (t, u_t_array, c_t_array, p_t_array, x_array, L_array)
    end

    res
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
    c_t_array::Vector{Array{Float64, 2}}
    p_t::Vector{Float64}
    x_array::Vector{Vector{Float64}}
    L_array::Vector{Float64}
    n_seed::Int
end


function simulate_all(args, io)



    Q = Int64(args["Q"])


    start::Int64 = Int64(args["start"])
    n::Int64 = Int64(args["n"])
    @show(n)
    n_args = copy(args)

    solution_output = Vector{Vector{SimResult}}()
    for i in 1:Threads.nthreads()
        push!(solution_output, Vector{SimResult}())
    end

    Threads.@threads for i in start:start+(n-1)
        L_array, x_array, u_array, u0_array = simulate_multi(i, n_args, Lm[1:Q], Ls[1:Q])
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
        t, u_t_array, c_t_array, p_t_array, x_array, L_array = simulate_multi(i, n_args, Lm[1:Q], Ls[1:Q]; tc=true)
        s = SimResult_TC(t, u_t_array, c_t_array, p_t_array, x_array, L_array, i)
        push!(solution_output[Threads.threadid()], s)
    end

    solution_output = vcat(solution_output...)
    solution_seed = [s.n_seed for s in solution_output]
    solution_output = solution_output[sortperm(solution_seed)]

    for i in 1:length(solution_output)
        x_array = solution_output[i].x_array
        u_t_array = solution_output[i].u_t_array
        c_t_array = solution_output[i].c_t_array
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
            for j in 1:length(t)
                print(io, join(map(string, c_t_array[q][j,:]), ',') * '\n')
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
            default=100 # (No unit)
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
            default=1.0 # (a.u.)
            help="threshold for HEI10 uptake"
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

        "--a_nodes"
            arg_type=Float64   # (s^{-1})
            default=0.14       # Rate of HEI10 absorption from pool to RIs


         "--beta_2"
            arg_type=Float64   # (s^{-1})
            default=0.03        # RI HEI10 escape rate into pool (C-dependent)

        "--beta_3"
            arg_type=Float64   # (s^{-1})
            default=0.0        # RI HEI10 escape rate into pool (const) - This is zero in all the simulations in the paper

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
        "--a_SC"
            arg_type=Float64
            default=0.0 #0.2*0.1

        "--b_SC"
            arg_type=Float64
            default=0.0 # 0.2*0.1

        "--pool_decay"
            arg_type=Float64
            default=0.0

        "--Q"
            arg_type=Int64
            default=5          # Number of chromosomes


        "--use_poisson"
            arg_type=Bool      # Whether to distribute RIs at a constant rate per uniform length
            default=true      # (rather than a number proportional to chromosome length)

        "--M_tot_mean"
            arg_type=Float64
            default=1170.0 # 237.2*(1.2+0.5*6.8 + 2*0.5*1.0*0.1*0.5*6.8  )
            help="mean total initial HEI10 amount"

	 "--M_tot_sd"
            arg_type=Float64
            default=75.0 
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


function plot_obj(filename, n_args; tc=false)

    println("Arguments")
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


function plot_recycling_survey_ends(filename, p; n=1000, ts=100, tc=false)
    args = parse_commandline()

    # Update simulation parameters from function call parameters
    n_args = copy(args)


    n_args["n"]  = n
    n_args["a_SC"] = 0.0
    n_args["b_SC"] = 0.0

    n_args["a_nodes"] *= p[1]
    n_args["beta_2"] *= p[2]
    n_args["alpha"] *= p[3]
    n_args["beta"] *= p[4]
    n_args["D"] *= p[5]


    n_args["M_tot_sd"]=p[6]*sqrt(p[7]^2*(n_args["M_tot_mean"]^2 + n_args["M_tot_sd"]^2) + n_args["M_tot_sd"]^2)
    n_args["M_tot_mean"] *= p[6]
    
    
    n_args["K_u"] = p[8]

    n_args["n_ts"] = ts

    plot_obj(filename, n_args; tc=tc)
end

s = 1.0
k = 1.0
r = 0.0
plot_recycling_survey_ends("../output/simulation_output/survey_wt_" * string(r) * "_" * string(s) * ".dat", [r, r, s, s, 1, 1, 0.0, 0.5]; n=1000)#, tc=true)

s = 1.0
k = 1.0
r = 0.0
plot_recycling_survey_ends("../output/simulation_output/survey_wt_ox_" * string(r) * "_" * string(s) * ".dat", [r, r, s, s, 1, 4.5, 0.0, 0.5]; n=1000)#, tc=true)

for r in [0.1]
    s = 1-r
    k = 1.0
    plot_recycling_survey_ends("../output/simulation_output/survey_interpolate_ends_" * string(r) * "_" * string(s) * ".dat", [r, r, s, s, 1, 1, 0.0, 0.5]; n=1000)#, tc=true)
end

for r in [0.1]
    s = 1-r
    k = 1.0
    plot_recycling_survey_ends("../output/simulation_output/survey_interpolate_ends_ox_" * string(r) * "_" * string(s) * ".dat", [r, r, s, s, 1, 4.5, 0.0, 0.5]; n=1000)#, tc=true)
end
