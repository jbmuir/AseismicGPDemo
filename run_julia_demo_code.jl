import Pkg
Pkg.Registry.add("https://github.com/jbmuir/JBMuirJuliaRegistry.jl")
Pkg.activate(".")
Pkg.instantiate()

include("aseismicgp_runsyntheticdata.jl")
include("aseismicgp_plotsyntheticdata.jl")
include("aseismicgp_plotsquarelengthscales.jl")

include("aseismicgp_runridgecrestdata.jl")
include("aseismicgp_plotridgecrestdata.jl")

include("aseismicgp_runcahuilladata.jl")
include("aseismicgp_plotcahuilladata.jl")