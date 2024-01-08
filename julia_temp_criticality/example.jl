using Pkg;
Pkg.activate("./")


using Plots

# this code will run on a personal computer, assuming you have 4 cores available for computations.
# if you have more cores, you can change the number of workers below to increase the parallel performance, and then you can also increase
# the values of n and T to get better statistics

using Distributed;
# add 4 workers for better parallel performance
addprocs(4);

@everywhere using julia_temp_criticality;


function run_simulation(n, k, T, Bs)
    results = pmap(B -> order_parameter_lstsq(meanfield_simulation(n, k, T, B)), Bs)
    return results
end


Bs = 2.5:0.05:4.5
k = 7
n = 10_000
T = 1000

simulations = run_simulation(n, k, T, Bs);

# we approximate Bc as 4 for simplicity

pseudo_theory = [positive_part.(4 - B) for B in Bs];


p = plot(Bs, simulations, label = "simulation data", xlabel = "B", ylabel = "order parameter V", title = "mean field simulation", legend = :topleft)
plot!(Bs, pseudo_theory, label = "approximate theory")

# show figure
savefig(p, "./example.png")
