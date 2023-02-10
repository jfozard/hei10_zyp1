
include("common_pch2_both.jl")


function plot_survey(filename, p; n=100, ts=100, tc=false)
    args = parse_commandline()
    println("Parsed args:")

    # Update simulation parameters from function call parameters
    n_args = copy(args)
    n_args["a_nodes"] *= p[1]
    n_args["a_SC"] = 0.0
    n_args["b_SC"] = 0.0
    n_args["beta_2"] *= p[1]
    n_args["alpha"] *= p[2]
    n_args["beta"] *= p[3]
    n_args["D"] *= p[4]


    n_args["M_tot_sd"] = p[5]*n_args["M_tot_sd"]
    n_args["M_tot_mean"] *= p[5]

    n_args["K_u"] = p[6]
    n_args["n_ts"] = ts
    n_args["n"] = n
    n_args["regtot"] = true
    plot_obj(filename, n_args; tc=tc)
end



@show("done load ends")



r = 0.2
s = 0.15
t = 0.1
plot_survey("../output/simulation_output/pch_" * string(r)  * "_" * string(s) * "_"* string(t) * "_regtot.dat", [t, r, s, 1.0, 1.0, 0.5]; n=1000)