import Pkg
Pkg.Registry.add(Pkg.RegistrySpec(url = "https://github.com/jbmuir/JBMuirJuliaRegistry.git"))
Pkg.activate(".")
Pkg.instantiate()

using AseismicGP

function smooth_catalog_rate(catalog::Catalog, n_half_smooth=20)
    t = catalog.t
    r = similar(t)
    for i in eachindex(r)
        il = max(1,i-n_half_smooth) # assume indices start at one, which is ok for the AseismicGP codebase (but not in general for Julia)
        iu = min(length(r), i+n_half_smooth)
        t_span_i = t[iu]-t[il]
        r[i] = (2*n_half_smooth + 1)/t_span_i
    end
    return r
end

include("aseismicgp_runsyntheticdata.jl")
include("aseismicgp_plotsyntheticdata.jl")
include("aseismicgp_plotsquarelengthscales.jl")

include("aseismicgp_runridgecrestdata.jl")
include("aseismicgp_plotridgecrestdata.jl")

include("aseismicgp_runcahuilladata.jl")
include("aseismicgp_plotcahuilladata.jl")
