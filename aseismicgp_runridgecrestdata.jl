using AseismicGP
using MCMCChains
using Distributions
using Optim
using DelimitedFiles
using Turing
using Random
using JLD2
using Dates

ridgecrest_data = reverse(readdlm("Data/ridgecrest_data.txt"), dims=1)
ridgecrest_min_mag = 3.0 #Arbitrary cutoff so that we don't have 10k+ events
ridgecrest_elapsed_time = 14.0 #days
ridgecrest_start_date = DateTime(2019,7,1)
ridgecrest_start_time = 18078.0 #obspy.matplotlib_date
ridgecrest_catalog = Catalog(ridgecrest_data[:,1].-ridgecrest_start_time,
                           ridgecrest_data[:,2],
                           ridgecrest_min_mag,
                           ridgecrest_start_date,
                           ridgecrest_elapsed_time)


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
αa, αb = inverse_gamma_tuner(0.5, 2.5) # suitable ranges found by using the etasap function from SAPP 
ca, cb = inverse_gamma_tuner(0.01, 0.1) # for maximum likelihood estimation on the Mw 7.1 aftershock tail
p̃a, p̃b = inverse_gamma_tuner(0.1, 1.0)

# Random.seed!(43771120)
N=129

etaspriors = ETASPriors(truncated(Normal(0,0.5),0,1.0), 
                        InverseGamma(αa, αb), 
                        InverseGamma(ca, cb), 
                        InverseGamma(p̃a, p̃b))				

crpt = ConstantRateParameters(ridgecrest_elapsed_time, μa, μb, etaspriors) 
spde1priors = ScalarSPDELayerPriors(Gamma(μa, μθ), Uniform(5, 10), truncated(Normal(0,1),0,Inf))
olrp = OneLayerRateParameters(ridgecrest_elapsed_time, N, spde1priors, etaspriors)
spde1priors2 = ScalarSPDELayerPriors(Uniform(5,10), Uniform(5,10), truncated(Normal(0,1),0,Inf))
spde2priors2 = VectorSPDELayerPriors(truncated(Gamma(μa, μθ),0,0.03), truncated(Normal(0,1),0,Inf))
tlrp = TwoLayerRateParameters(ridgecrest_elapsed_time, N, spde1priors2, spde2priors2, etaspriors)

# for (model_label, model) in zip(["zero", "one", "two"], [crpt, olrp, tlrp])
#     etasm, etasc = etas_sampling(100_000, 6, ridgecrest_catalog, model, threads=true)
#     jldsave("Outputs/ridgecrest_$(model_label).jld2"; etasm, etasc)
# end
for (model_label, model) in zip(["zero"], [crpt])
    Random.seed!(43771120)
    etasm, etasc = etas_sampling(100_000, 6, ridgecrest_catalog, model, threads=true)
    jldsave("Outputs/ridgecrest_$(model_label).jld2"; etasm, etasc)
end
#