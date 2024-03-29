using AseismicGP
using MCMCChains
using Distributions
using Optim
using StatsPlots
using Plots
using Turing
using Random
using JLD2
using CSV
using CairoMakie

gr()

const μtrue = 0.2
const Ktrue = 0.2
const αtrue = 1.25
const ptrue = 1.2
const ctrue = 0.05
const tspan = 1000.0
const bvalue = 1.0
const N = 129

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

Ka, Kb = inverse_gamma_tuner(Ktrue/2, 2*Ktrue)
αa, αb = inverse_gamma_tuner(αtrue/2, 2*αtrue)
ca, cb = inverse_gamma_tuner(ctrue/2, 2*ctrue)
p̃a, p̃b = inverse_gamma_tuner((ptrue-1)/2, 2*(ptrue-1))

f_const(t) = μtrue
f_gauss(t) = 4.8 * μtrue * exp(-(t-tspan/2)^2/(2*tspan^2/64)) + 0.2*μtrue
f_square(t) = (tspan/4 < t < 3*tspan/4) ? 5*μtrue : 0.2*μtrue

Random.seed!(43771120)

cat_gen_const = ETASInhomogeneousPP(f_const, Ktrue, αtrue, ctrue, ptrue)
cat_const = simulate_ETAS(cat_gen_const, tspan, bvalue)

cat_gen_gauss = ETASInhomogeneousPP(f_gauss, Ktrue, αtrue, ctrue, ptrue)
cat_gauss  = simulate_ETAS(cat_gen_gauss , tspan, bvalue)

cat_gen_square = ETASInhomogeneousPP(f_square, Ktrue, αtrue, ctrue, ptrue)
cat_square = simulate_ETAS(cat_gen_square, tspan, bvalue)

etaspriors = ETASPriors(truncated(Normal(0,Ktrue),0,2*Ktrue), 
                        InverseGamma(αa, αb), 
                        InverseGamma(ca, cb), 
                        InverseGamma(p̃a, p̃b))				


for (cat_label,catalog, fun) in zip(["const", "gauss", "square"], [cat_const, cat_gauss, cat_square], [f_const, f_gauss, f_square])
    for (cat_label,catalog, fun) in zip(["const", "gauss", "square"], [cat_const, cat_gauss, cat_square], [f_const, f_gauss, f_square])
    μa, μb, μθ = gamma_moment_tuner(length(catalog)/tspan, length(catalog)/tspan/2)
    crpt = ConstantRateParameters(tspan, μa, μb, etaspriors) 
    spde1priors = ScalarSPDELayerPriors(Gamma(μa, μθ), Uniform(10, 300), truncated(Normal(0,1),0,Inf))
    olrp = OneLayerRateParameters(tspan, N, spde1priors, etaspriors)
    spde1priors2 = ScalarSPDELayerPriors(Uniform(10,200), Uniform(100,300), truncated(Normal(0,1),0,Inf))
    spde2priors2 = VectorSPDELayerPriors(truncated(Gamma(μa, μθ),0,2*length(catalog)/tspan), truncated(Normal(0,1),0,Inf))
    tlrp = TwoLayerRateParameters(tspan, N, spde1priors2, spde2priors2, etaspriors)

    for (model_label, model) in zip(["zero", "one", "two"], [crpt, olrp, tlrp])
        etasm = load("Outputs/$(cat_label)_$(model_label).jld2", "etasm")
        etasc = load("Outputs/$(cat_label)_$(model_label).jld2", "etasc")
        etasc_params = Turing.MCMCChains.Chains(etasc[50_001:10:end], filter(x->x!=:logposterior, sections(etasc)))
        quantities = generated_quantities(etasm, etasc_params)

        if model_label == "two"
            μ = hcat([q[1] for q in quantities]...)
            μl = hcat([q[2] for q in quantities]...)
        else
            μ = hcat([q for q in quantities]...)
        end

        summary = summarize(etasc)
        CSV.write("Outputs/chainsummary_$(cat_label)_$(model_label).csv", summary)

        Plots.plot(etasc[50_001:10:end], lw=2)
        savefig("Figures/chainplot_$(cat_label)_$(model_label).pdf")

        m05 = [quantile(μi, 0.05) for μi in eachrow(μ)]
        m25 = [quantile(μi, 0.25) for μi in eachrow(μ)]
        m50 = [quantile(μi, 0.50) for μi in eachrow(μ)]
        m75 = [quantile(μi, 0.75) for μi in eachrow(μ)]
        m95 = [quantile(μi, 0.95) for μi in eachrow(μ)]

        f = Figure()

        ax1 = Axis(f[1, 1], xlabel="Sequence Day", ylabel="Rate (Day⁻¹)")
        ax2 = Axis(f[1, 1], yaxisposition = :right, ylabel="Magnitude")
        CairoMakie.xlims!(ax1, [0,tspan])
        CairoMakie.xlims!(ax2, [0,tspan])
        hidedecorations!(ax1, ticks=false, ticklabels=false, label=false)
        hidexdecorations!(ax2)
        hideydecorations!(ax2, ticks=false, ticklabels=false, label=false)
    
        ev = CairoMakie.scatter!(ax2, catalog.t, catalog.M, color = (:black, 0.25))
    
        if length(m05) == 1
            band!(ax1,[0,tspan], [m05[1], m05[1]], [m95[1], m95[1]], color = (:blue, 0.35))
            band!(ax1,[0,tspan], [m25[1], m25[1]], [m75[1], m75[1]], color = (:blue, 0.35))
            rmu = lines!(ax1,[0,tspan], [m50[1], m50[1]], label="Posterior μ", color=:blue, linewidth=2)
        else
            band!(ax1,0:model.M.h:tspan, m05, m95, color = (:blue, 0.35))
            band!(ax1,0:model.M.h:tspan, m25, m75, color = (:blue, 0.35))
            rmu = lines!(ax1,0:model.M.h:tspan, m50, color=:blue, linewidth=2)
        end

        rfun = lines!(ax1, 0:(tspan/1000):tspan, fun.(0:(tspan/1000):tspan), color=:red, linewidth=2)

        axislegend(ax1, [ev, rmu, rfun], ["Events", "Posterior μ", "True μ"], nothing, position = :rt)
    
        save("Figures/muplot_$(cat_label)_$(model_label).pdf", f)

    end
end                   
#
