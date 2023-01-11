using AseismicGP
using MCMCChains
using Distributions
using Optim
using Turing
using Random
using JLD2

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

for (cat_label,catatalog, f) in zip(["const", "gauss", "square"], [cat_const, cat_gauss, cat_square], [f_const, f_gauss, f_square])
    μa, μb, μθ = gamma_moment_tuner(length(catatalog)/tspan, length(catatalog)/tspan/2)
    crpt = ConstantRateParameters(tspan, μa, μb, etaspriors) 
    spde1priors = ScalarSPDELayerPriors(Gamma(μa, μθ), Uniform(150, 250), truncated(Normal(0,1),0,Inf))
    olrp = OneLayerRateParameters(tspan, N, spde1priors, etaspriors)
    spde1priors2 = ScalarSPDELayerPriors(Uniform(50,150), Uniform(150,250), truncated(Normal(0,1),0,Inf))
    spde2priors2 = VectorSPDELayerPriors(truncated(Gamma(μa, μθ),0,2*length(catatalog)/tspan), truncated(Normal(0,1),0,Inf))
    tlrp = TwoLayerRateParameters(tspan, N, spde1priors2, spde2priors2, etaspriors)

    for (model_label, model) in zip(["zero", "one", "two"], [crpt, olrp, tlrp])
        Random.seed!(43771120)
        println("Running synthetic $cat_label $model_label")
        @time etasm, etasc = etas_sampling(100_000, 6, catatalog, model, threads=true)
        jldsave("Outputs/$(cat_label)_$(model_label).jld2"; etasm, etasc)
    end
end                   
#