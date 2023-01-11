using AseismicGP
using MCMCChains
using Distributions
using Optim
using DelimitedFiles
using Turing
using Random
using JLD2
using Dates
using CSV
using StatsPlots
using Plots
using CairoMakie

ridgecrest_data = reverse(readdlm("Data/ridgecrest_data.txt"), dims=1)
ridgecrest_min_mag = 3.0 #SCSN completeness magnitude estimate
ridgecrest_elapsed_time = 14.0 #days
ridgecrest_start_date = DateTime(2019,7,1)
ridgecrest_start_time = 18078.0 #obspy.matplotlib_date
ridgecrest_catalog = Catalog(ridgecrest_data[:,1].-ridgecrest_start_time,
                           ridgecrest_data[:,2],
                           ridgecrest_min_mag,
                           ridgecrest_start_date,
                           ridgecrest_elapsed_time)

smooth_rate = smooth_catalog_rate(ridgecrest_catalog)

function gamma_moment_tuner(μ, σ)
    # gives parameters for a Gamma with given mean and standard deviation
    a = (μ/σ)^2
    b = μ/σ^2
    θ = 1/b
    return (a, b, θ)
end

function inverse_gamma_tuner(l, u; p=0.01)
    # Tunes an inverse gamma to have p percentile mass below l and above u
    function f(θ)
        a, b = exp.(θ)
        l_cp_res = cdf(InverseGamma(a,b), l) - p
        u_cp_res = 1-cdf(InverseGamma(a,b), u) - p
        l_cp_res^2 + u_cp_res^2
    end
    
    res = optimize(f, zeros(2))
    a, b = exp.(res.minimizer)
    return (a, b)
end

μa, μb, μθ = gamma_moment_tuner(0.02, 0.01) #rough estimate from background rate 2000-2019
αa, αb = inverse_gamma_tuner(0.5, 3.0)
ca, cb = inverse_gamma_tuner(0.01, 0.1)
p̃a, p̃b = inverse_gamma_tuner(0.1, 0.5)

N=129

etaspriors = ETASPriors(truncated(Normal(0,0.25),0,0.5), 
                        InverseGamma(αa, αb), 
                        InverseGamma(ca, cb), 
                        InverseGamma(p̃a, p̃b))				

crpt = ConstantRateParameters(ridgecrest_elapsed_time, μa, μb, etaspriors) 
spde1priors = ScalarSPDELayerPriors(Gamma(μa, μθ), Uniform(150, 250), truncated(Normal(0,1),0,Inf))
olrp = OneLayerRateParameters(ridgecrest_elapsed_time, N, spde1priors, etaspriors)
spde1priors2 = ScalarSPDELayerPriors(Uniform(50,150), Uniform(150,250), truncated(Normal(0,1),0,Inf))
spde2priors2 = VectorSPDELayerPriors(truncated(Gamma(μa, μθ),0,0.02), truncated(Normal(0,1),0,Inf))
tlrp = TwoLayerRateParameters(ridgecrest_elapsed_time, N, spde1priors2, spde2priors2, etaspriors)

for (model_label, model) in zip(["zero", "one", "two"], [crpt, olrp, tlrp])
    etasm = load("Outputs/ridgecrest_$(model_label).jld2", "etasm")
    etasc = load("Outputs/ridgecrest_$(model_label).jld2", "etasc")
    etasc_params = Turing.MCMCChains.Chains(etasc[50_001:10:end], filter(x->x!=:logposterior, sections(etasc)))
    quantities = generated_quantities(etasm, etasc_params)

    if model_label == "two"
        μ = hcat([q[1] for q in quantities]...)
    else
        μ = hcat([q for q in quantities]...)
    end

    summary = summarize(etasc)
    CSV.write("Outputs/chainsummary_ridgecrest_$(model_label).csv", summary)

    Plots.plot(etasc[50_001:10:end], lw=2)
    savefig("Figures/chainplot_ridgecrest_$(model_label).pdf")

    m05 = [quantile(μi, 0.15) for μi in eachrow(μ)]
    m25 = [quantile(μi, 0.25) for μi in eachrow(μ)]
    m50 = [quantile(μi, 0.50) for μi in eachrow(μ)]
    m75 = [quantile(μi, 0.75) for μi in eachrow(μ)]
    m95 = [quantile(μi, 0.95) for μi in eachrow(μ)]

    f = Figure()

    ax1 = Axis(f[1, 1], xlabel="Sequence Day", ylabel="Rate (Day⁻¹)")
    ax2 = Axis(f[1, 1], yaxisposition = :right, ylabel="Magnitude")
    hidedecorations!(ax1, ticks=false, ticklabels=false, label=false)
    hidexdecorations!(ax2)
    hideydecorations!(ax2, ticks=false, ticklabels=false, label=false)

    ev = CairoMakie.scatter!(ax2, ridgecrest_catalog.t, ridgecrest_catalog.M, color = (:black, 0.25))

    if length(m05) == 1
        band!(ax1,[0,ridgecrest_elapsed_time], [m05[1], m05[1]], [m95[1], m95[1]], color = (:blue, 0.35))
        band!(ax1,[0,ridgecrest_elapsed_time], [m25[1], m25[1]], [m75[1], m75[1]], color = (:blue, 0.35))
        rmu = lines!(ax1,[0,ridgecrest_elapsed_time], [m50[1], m50[1]], label="Posterior μ", color=:blue, linewidth=2)
    else
        band!(ax1,0:model.M.h:ridgecrest_elapsed_time, m05, m95, color = (:blue, 0.35))
        band!(ax1,0:model.M.h:ridgecrest_elapsed_time, m25, m75, color = (:blue, 0.35))
        rmu = lines!(ax1,0:model.M.h:ridgecrest_elapsed_time, m50, color=:blue, linewidth=2)
    end

    sr = lines!(ax1, ridgecrest_catalog.t, smooth_rate, color = :black)

    axislegend(ax1, [ev, rmu, sr], ["Events", "Posterior μ", "Smoothed Total Rate"], nothing, position = :rt)


    save("Figures/muplot_ridgecrest_$(model_label).pdf", f)

end
#