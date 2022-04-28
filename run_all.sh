set -e
julia=~/JuliaPro-1.4.0-1/Julia/bin/julia
export JULIA_NUM_THREADS=6
$julia zyp_output.jl
$julia kymo_sol.jl
python all_path_analysis.py
python all_wt_analysis.py
python expt_univalents.py
python f_plot.py
python julia_plots_zyp1.py
python plot_beta.py
python plot_kymo.py
python plot_mlh.py
python new_fig2.py
python new_fig4.py  
