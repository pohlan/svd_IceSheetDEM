# Assumes that the name of the files are all fixed

using svd_IceSheetDEM

const F = Float32 # Julia default is Float64 but that kills the process for the full training data set if r is too large

ARGS = ["--lambda", "1e5",
        "--r", "377",
        "--imbie_path", "data/gris-imbie-1980/",
        "--train_folder", "data/training_data_it0_1200/"]
parsed_args = parse_commandline(ARGS)

imbie_path         = parsed_args["imbie_path"]
aerodem_path       = "data/aerodem/"
bedmachine_path    = "data/bedmachine/"
training_data_path = parsed_args["train_folder"]

# 1.) make sure the training data set is not empty
@assert !isempty(training_data_path)
## choose a template file to make saving of netcdf files easier
template_path      = training_data_path*readdir(training_data_path)[1]
## derive grid size in m from training data
const gr = parse(Int, split(template_path, "_")[end-6][2:end-1])

# 2.) move all the necessary bedmachine layers to the right grid (downloads the bedmachine-v5 if not available)
bedmachine_file = bedmachine_path * "bedmachine_g$(gr).nc"
if !isfile(bedmachine_file)
    create_bedmachine_grid(gr; bedmachine_path, template_path)
end

# 3.) check if aerodem is available at the right grid, if not warp from available one or download/create from scratch
aerodem_g150 = aerodem_path * "aerodem_rm-filtered_geoid-corr_g150.nc"
obs_file     = aerodem_path*"aerodem_rm-filtered_geoid-corr_g$(gr).nc"
if !isfile(aerodem_path * "aerodem_rm-filtered_geoid-corr_g$(gr).nc")
    if !isfile(aerodem_g150)
        # create aerodem, for some reason the cutting with the shapefile outline only works for smaller grids
        # otherwise GDALError (CE_Failure, code 1): Cutline polygon is invalid.
        create_aerodem(aerodem_path, imbie_path, bedmachine_path)
    end
    gdalwarp(aerodem_g150; grid=gr, srcnodata="0.0", dest=obs_file)
end

# 4.) make sure that the imbie shp file is downloaded and get a netcdf mask of the right grid
imbie_mask = imbie_path * "imbie_mask_g$(gr).nc"
if !isfile(imbie_path * "gris-outline-imbie-1980.shp")
    @error "shape file of imbie outline not downloaded or not in a local folder data/gris-imbie-1980/"
end
if !isfile(imbie_mask)
    create_imbie_mask(gr; imbie_path, sample_path=aerodem_g150)
end


# 4.) run the svd solve_lsqfit

# 377 -> findfirst(cumsum(Σ)./sum(Σ).>0.9)
# retrieve command line arguments
λ           = F(parsed_args["λ"])     # regularization
r           = parsed_args["r"]
rec_file = solve_lsqfit(F, λ, r, gr, imbie_mask, training_data_path, obs_file)

# 5.) calculate the floating mask
create_reconstructed_bedmachine(obs_file, rec_file, bedmachine_file, template_path)
