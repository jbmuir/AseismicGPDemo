#!/bin/bash

export JULIA_THREADS=8

mkdir Figures
mkdir Outputs
#uncomment the below lines if you want to download the data, but they are included in the repository
#python3 Data/aseismicgp_downloaddata_cahuilla.py
#python3 Data/aseismicgp_downloaddata_ridgecrest.py
julia -t $JULIA_THREADS run_julia_demo_code.jl
