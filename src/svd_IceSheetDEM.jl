__precompile__(false)
module svd_IceSheetDEM

using ArgParse

import ArchGDAL as AG
import GeoFormatTypes as GFT

using DelimitedFiles, NCDatasets, NetCDF, Glob, DataFrames, CSV, Dates, ZipFile
using Downloads, Cascadia, Gumbo, HTTP
using Printf, ProgressMeter
using Statistics, GeoStats, StatsBase, Distributions, Interpolations, LsqFit, ImageFiltering, Distances, ParallelRandomFields.grf2D_CUDA
using Arpack, LinearAlgebra
using DataStructures: OrderedDict
import Plots, StatsPlots

export parse_commandline
export archgdal_read, gdalwarp
export create_aerodem, create_bedmachine_grid, create_imbie_mask, create_atm_grid, create_dhdt_grid
export do_reconstruction, create_reconstructed_bedmachine
export residual_analysis

function parse_commandline(args)
    s = ArgParseSettings()
    @add_arg_table s begin
        "--λ", "--lambda"
            help     = "regularization parameter for the least squares problem"
            arg_type = Float64
            default  = 1e5
        "--r"
            help     = "truncation of SVD, default is a full SVD"
            arg_type = Int
            default  = 10^30  # numbers close to the maximum or larger will give a full SVD
        "--training_data"
            help     = "training files, e.g. train_folder/usurf*.nc"
            nargs    = '*'
            arg_type = String
            required = true
        "--shp_file"
            help     = "shape file outlining the ice"
            arg_type = String
            required = true
        "--do_figures"
            help     = "whether or not to plot the difference of the reconstructed elevations to aerodem data (plotting and saving requires a bit of extra memory)"
            arg_type = Bool
            default  = false
        "--use_arpack"
            help     = "If this is set to true, then the Arpack svd is used instead of the standard LinearAlgebra algorithm. Arpack is iterative and matrix free and thus useful when memory becomes limiting, but it can be slower."
            arg_type = Bool
            default  = false
    end
    return parse_args(args,s)
end

include("gdal_helpers.jl")
include("reconstruction_routines.jl")
include("statistics_helpers.jl")
include("model_selection.jl")

end # module svd_IceSheetDEM
