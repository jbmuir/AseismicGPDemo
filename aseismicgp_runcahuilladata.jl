using AseismicGP
using MCMCChains
using Distributions
using Optim
using DelimitedFiles
using Turing
using Random
using JLD2
using Dates

cahuilla_data = reverse(readdlm("Data/cahuilla_data.txt"), dims=1)
cahuilla_min_mag = 1.71 #SCSN completeness magnitude estimate
cahuilla_elapsed_time = 1460.0 #days
cahuilla_start_date = DateTime(2016,1,1)
cahuilla_start_time = 16801.0 #obspy.matplotlib_date
cahuilla_catalog = Catalog(cahuilla_data[:,1].-cahuilla_start_time,
                           cahuilla_data[:,2],
                           cahuilla_min_mag,
                           cahuilla_start_date,
                           cahuilla_elapsed_time)


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

μa, μb, μθ = gamma_moment_tuner(0.01, 0.005) #rough estimate from background rate 2000-2016
αa, αb = inverse_gamma_tuner(0.5, 3.0)
ca, cb = inverse_gamma_tuner(0.01, 0.1)
p̃a, p̃b = inverse_gamma_tuner(0.1, 0.5)

N=129

etaspriors = ETASPriors(truncated(Normal(0,0.25),0,0.5), 
                        InverseGamma(αa, αb), 
                        InverseGamma(ca, cb), 
                        InverseGamma(p̃a, p̃b))				

crpt = ConstantRateParameters(cahuilla_elapsed_time, μa, μb, etaspriors) 
spde1priors = ScalarSPDELayerPriors(Gamma(μa, μθ), Uniform(150, 250), truncated(Normal(0,1),0,Inf))
olrp = OneLayerRateParameters(cahuilla_elapsed_time, N, spde1priors, etaspriors)
spde1priors2 = ScalarSPDELayerPriors(Uniform(50,150), Uniform(150,250), truncated(Normal(0,1),0,Inf))
spde2priors2 = VectorSPDELayerPriors(truncated(Gamma(μa, μθ),0,0.02), truncated(Normal(0,1),0,Inf))
tlrp = TwoLayerRateParameters(cahuilla_elapsed_time, N, spde1priors2, spde2priors2, etaspriors)

for (model_label, model) in zip(["zero", "one", "two"], [crpt, olrp, tlrp])
    Random.seed!(43771120)
    println("Running cahuilla $model_label")
    @time etasm, etasc = etas_sampling(100_000, 6, cahuilla_catalog, model, threads=true)
    jldsave("Outputs/cahuilla_$(model_label).jld2"; etasm, etasc)
end
#