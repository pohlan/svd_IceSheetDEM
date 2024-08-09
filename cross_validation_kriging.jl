using svd_IceSheetDEM, NetCDF

shp_file      = "data/gris-imbie-1980/gris-outline-imbie-1980_updated.shp"
template_file = "data/training_data_it2_600/usurf_ex_gris_g600m_v2023_GIMP_id_0_1980-1-1_2020-1-1.nc"
const gr = 600

# files
bedm_file                   = create_bedmachine_grid(gr, template_file)
bedmachine_path             = splitdir(bedm_file)[1]
aerodem_g150, obs_aero_file = create_aerodem(;gr, shp_file, bedmachine_path)
obs_ATM_file                = get_atm_file()
dhdt_file, _                = create_dhdt_grid(;gr, startyr=1994, endyr=2010)
mask_file                   = create_imbie_mask(;gr, shp_file, sample_path=aerodem_g150)

main_output_dir  = joinpath("output","geostats_interpolation")
fig_path         = joinpath(main_output_dir, "figures/")
atm_dh_dest_file = joinpath(dirname(obs_ATM_file), "grimp_minus_atm.csv")
mkpath(fig_path)

# number of bins
nbins1 = 6
nbins2 = 10

grid_output, geotable, varg, get_predicted_field = svd_IceSheetDEM.do_destandardization(bedm_file, bedm_file, obs_aero_file, obs_ATM_file, dhdt_file, mask_file, fig_path, atm_dh_dest_file; nbins1, nbins2)

###############
# choose maxn #
###############
maxn = 40

output_path = joinpath(main_output_dir, "kriging/")
mkpath(output_path)
println("Kriging...")
interp = svd_IceSheetDEM.do_kriging(grid_output, geotable, varg; maxn)

h_predict = get_predicted_field(interp.Z)

dest_file                    = joinpath(output_path, "rec_kriging_maxn$(maxn).nc")
svd_IceSheetDEM.save_netcdf(dest_file, obs_aero_file, [h_predict], ["surface"], Dict("surface" => Dict{String,Any}()))


obs = ncread(obs_aero_file, "Band1")
iobs = findall(obs .!= -9999 .&& h_predict .!= -9999)

df = zeros(size(obs))
df[iobs] .= obs[iobs] .- h_predict[iobs]
