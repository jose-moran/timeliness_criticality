using Distributions
using StatsBase
using FFTW

export meanfield_simulation,
    get_mean_diff,
    get_order_parameter,
    detrend,
    power_analysis,
    power_analysis_lstsq,
    get_fluctuations_lstsq,
    variogram_analysis,
    order_parameter_lstsq,
    positive_part

"""
    positive_part(x)
return x if x is positive, 0 otherwise
"""
function positive_part(x)
    (x < 0 ? 0 : x)
end

"""
    meanfield_simulation(n::Int64, d::Int64, T::Int64)
run mean-field simulation with n nodes and d neighbours
"""
function meanfield_simulation(n::Int64, d::Int64, T::Int64, B)
    taus = Array{Float64}(undef, n, T)
    eps = rand(Exponential(), (n, T))
    taus[:, 1] = eps[:, 1]
    for t ∈ 2:T
        for i ∈ 1:n
            taus[i, t] = positive_part(maximum(taus[rand(1:n, d), t-1]) - B) + eps[i, t]
        end
    end
    return taus
end

"""
    random_regular_digraph(n::Int64, d::Int64)
return a simulation on a random regular directed graph with n nodes and d neighbors
"""
function directed_graph_simulation(n::Int64, d::Int64, T::Int64, B)
    # define a directed erdos renyi graph with n nodes and d neighbors
    G = random_regular_digraph(n, d)
    taus = Array{Float64}(undef, n, T)
    eps = rand(Exponential(), (n, T))
    taus[:, 1] = eps[:, 1]
    # store the neighbors of each node in a list of lists
    neigh = [neighbors(G, i) for i ∈ 1:n]
    for t ∈ 2:T
        for i ∈ 1:n
            # get the neighbors of the node i
            neighbors_i = neigh[i]
            taus[i, t] = positive_part(maximum(taus[neighbors_i, t-1]) - B) + eps[i, t]
        end
    end
    return taus
end

"""
    get_mean_diff(taus::Array{Float64, 2})
return the mean difference between consecutive taus, i.e. the average of taus[i,:] - taus[i-1,:]
"""
function get_mean_diff(taus)
    return vec(mean(diff(taus, dims = 2), dims = 1))
end

"""
    get_order_parameter(taus::Array{Float64, 2})
return the order parameter of the simulation, i.e. the mean of get_mean_diff(taus), computed with get_mean_diff
"""
function get_order_parameter(taus)
    return mean(get_mean_diff(taus))
end


"""
    order_parameter_lstsq(taus::Array{Float64, 2})
return the order parameter of the simulation, i.e. the mean of get_mean_diff(taus), computed with least squares by regressing
the mean of taus against the time t, e.g. mean_tau(t) = V * t + noise 
and returning V
"""
function order_parameter_lstsq(taus)
    vt = vec(mean(taus, dims = 1))
    T = length(vt)
    xs = collect(1:T)
    return sum(xs .* vt) / sum(xs .^ 2)
end

"""
    detrend(taus::Array{Float64, 2})
return the detrended taus, i.e. taus with the trend removed (the cross-sectional average)
"""
function detrend(taus)
    T = size(taus, 2)
    trend = get_order_parameter(taus) * (0:T-1)
    return transpose(transpose(taus) .- trend)
end


"""
    get_fluctuations(taus::Array{Float64, 2})
return the fluctuations of the simulation, i.e. the average of the detrended taus)
"""
function get_fluctuations(taus)
    trend = get_order_parameter(taus) * (0:size(taus, 2)-1)
    fluctuations = vec(mean(taus, dims = 1)) .- trend
    return fluctuations
end

"""
    get_fluctuations_lstsq(taus::Array{Float64, 2})
return the fluctuations of the simulation, i.e. the average of the detrended taus, by computing the trend with least squares
"""
function get_fluctuations_lstsq(taus)
    vt = vec(mean(taus, dims = 1))
    T = length(vt)
    xs = collect(1:T)
    trend = sum(xs .* vt) / sum(xs .^ 2) * xs
    fluctuations = vt .- trend
    return fluctuations
end


"""
    get_power_spectrum(fluctuations::Vector{Float64})
return the power spectrum of the fluctuations, i.e. the absolute value of the fft of the fluctuations
"""
function get_power_spectrum(fluctuations)
    F = fftshift(fft(fluctuations))
    t = length(fluctuations)
    F = F[t÷2+1:end]
    return abs.(F)
end

"""
    power_analysis(n::Int64, d::Int64, T::Int64, B::Float64)
return the power spectrum of the fluctuations of a simulation with n nodes, d neighbors, T time steps and buffer B
"""
function power_analysis(n::Int64, d::Int64, T::Int64, B::Float64)
    taus = mean_field_simulation(n, d, T, B)
    fluctuations = get_fluctuations(taus)
    power_spectrum = get_power_spectrum(fluctuations)
    return power_spectrum
end

"""
    power_analysis_lstsq(n::Int64, d::Int64, T::Int64, B::Float64)
return the power spectrum of the fluctuations of a simulation with n nodes, d neighbors, T time steps and buffer B, by computing the trend with least squares
"""
function power_analysis_lstsq(n::Int64, d::Int64, T::Int64, B::Float64)
    taus = mean_field_simulation(n, d, T, B)
    fluctuations = get_fluctuations_lstsq(taus)
    power_spectrum = get_power_spectrum(fluctuations)
    return power_spectrum
end

"""
    variogram(x::Vector{T}, h::Int)
return the variogram of x at a lag h, i.e. the average of (x[i+h] - x[i]) ^ 2
"""
function variogram(x::Vector{T}, h::Int) where {T <: AbstractFloat}
    v = mean((x[h+1:end] - x[1:end-h]) .^ 2)
    return v
end

"""
    variogram(x::Vector{T}, hs::Vector{Int})
return the variogram of x at lags hs, i.e. the average of (x[i+h] - x[i]) ^ 2 for h in hs
"""
function variogram(x::Vector{T}, hs::Vector{Int}) where {T <: AbstractFloat}
    return [variogram(x, h) for h in hs]
end
"""
    variogram_analysis(n::Int64, d::Int64, T::Int64, B::Float64, hs::Vector{Int})
return the variogram of the fluctuations of a simulation with n nodes, d neighbors, T time steps and buffer B, at lags hs
"""
function variogram_analysis(n::Int64, d::Int64, T::Int64, B::Float64, hs::Vector{Int})
    taus = mean_field_simulation(n, d, T, B)
    fluctuations = get_fluctuations(taus)
    v = variogram(fluctuations, hs)
    return v
end
