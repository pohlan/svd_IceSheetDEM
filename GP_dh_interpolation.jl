using svd_IceSheetDEM
using GaussianProcesses, StatsBase, NCDatasets, Interpolations
import Plots; Plots.pyplot()

gr = 6000

# load data
ds_aero = NCDataset("data/grimp_minus_aero_g$(gr).nc")
dh_aero = ds_aero["dh"][:,end:-1:1]
dh_atm  = NCDataset("data/grimp_minus_atm_g$(gr).nc")["dh"][:,end:-1:1]
ds_mask = NCDataset("data/gris-imbie-1980/imbie_mask_g$(gr).nc")["Band1"]
bedm_mask = NCDataset("data/bedmachine/bedmachine_g$(gr).nc")["mask"]
x  = ds_aero["x"]
y  = ds_aero["y"]
h_aero = NCDataset("data/aerodem/aerodem_g$(gr)_aligned_median.nc")["surface"][:,end:-1:1]
h_atm  = NCDataset("data/ATM/ATM_elevation_g$(gr)_aligned_median.nc")["surface"][:,end:-1:1]
GrIMP  = NCDataset("data/bedmachine/bedmachine_g$(gr).nc")["surface"]

# create array containing aerodem and atm data
idx_aero = findall(.!ismissing.(dh_aero) .&& .!ismissing.(ds_mask) .&& (bedm_mask .!= 1))
idx_atm  = findall(.!ismissing.(dh_atm)  .&& .!ismissing.(ds_mask) .&& (bedm_mask .!= 1) .&& ismissing.(dh_aero))

dh = Array{Union{eltype(dh_aero),Missing}}(missing, size(dh_aero))
dh[idx_aero] .= dh_aero[idx_aero]
dh[idx_atm]  .= dh_atm[idx_atm]

h  = Array{Union{eltype(dh_aero),Missing}}(missing, size(dh_aero))
h[idx_aero] .= h_aero[idx_aero]
h[idx_atm]  .= h_atm[idx_atm]

# only do GP for subset of the ice sheet (for now)
x_i    = 64:250
y_i    = 1:300
nx, ny = length(x_i), length(y_i)
Z      = dh[x_i,y_i,1]
E      = GrIMP[x_i,y_i]

# only take points within ice mask and where there are no missing data points
idx    = findall(.!ismissing.(Z) .&& .!ismissing.(E))
X      = Float64.(repeat(x[x_i], 1, ny)[idx])
Y      = Float64.(repeat(y[y_i]', nx, 1)[idx])
ELV    = Float64.(E[idx])
z      = Float64.(Z[idx])

i_to_delete = findall(abs.(z) .> 3 .* mad(z))
deleteat!(X, i_to_delete)
deleteat!(Y, i_to_delete)
deleteat!(idx, i_to_delete)
deleteat!(z, i_to_delete)
deleteat!(ELV, i_to_delete)

# Plot the original field
z_m = Array{Union{eltype(dh_aero),Missing}}(missing, (nx,ny))
z_m[idx] .= z
Plots.heatmap(x[x_i], y[y_i][end:-1:1], z_m', cmap=:bwr, clims=(-40,40)); Plots.savefig("aplot.png")

# binning, find trend of dh w.r.t. elevation
z_binned, ELV_bin_centers = svd_IceSheetDEM.bin_equal_sample_size(GrIMP[x_i,y_i][idx], z, 400)
itp = linear_interpolation(ELV_bin_centers, median.(z_binned),  extrapolation_bc = Line())
Plots.scatter(ELV_bin_centers, median.(z_binned))
ELV_plot = minimum(GrIMP[x_i,y_i][idx]):100:maximum(GrIMP[x_i,y_i][idx])
Plots.plot!(ELV_plot, itp(ELV_plot), label="linear interpolation")
Plots.savefig("cplot.png")

# detrend
z_detrend = z .- itp(GrIMP[x_i,y_i][idx])
# Plot the detrended field
z_d = Array{Union{eltype(dh_aero),Missing}}(missing, (nx,ny))
z_d[idx] .= z_detrend
Plots.heatmap(x[x_i], y[y_i][end:-1:1], z_d', cmap=:bwr, clims=(-40,40)); Plots.savefig("eplot.png")

# prepare GP
m = MeanZero()
kern  = Matern(5/2,[log(1e5), log(1e5), log(200)], 0.1)

# fit GP
gp = GP([X Y ELV]',z_detrend,m,kern, log(1.0))
println("optimize...")
optimize!(gp)

# predict
println("predict...")
idx_predict = findall(.!ismissing.(ds_mask[x_i,y_i]) .&& (bedm_mask[x_i,y_i] .!= 1) .&& .!ismissing.(GrIMP[x_i,y_i]))
x_predict   = repeat(x[x_i],  1, ny)[idx_predict]
y_predict   = repeat(y[y_i]', nx, 1)[idx_predict]
elv_predict =         GrIMP[x_i,y_i][idx_predict]
mus, sig    = predict_y(gp, [vec(x_predict) vec(y_predict) vec(elv_predict)]')
dh_pred     =  Array{Union{eltype(dh_aero),Missing}}(missing, (nx,ny))
dh_pred[idx_predict] .= mus .+ itp(elv_predict)
Plots.heatmap(x[x_i], y[y_i][end:-1:1], dh_pred', cmap=:bwr, clims=(-30,30), axes=:equal); Plots.savefig("bplot.png")

# calculate final h
h_rec = Array{Union{eltype(dh_aero),Missing}}(missing, size(GrIMP))
h_rec[x_i, y_i] .= GrIMP[x_i,y_i] .- dh_pred
h_rec[idx_aero] .= h_aero[idx_aero]
Plots.heatmap(h_rec'); Plots.savefig("eplot.png")

# save as netcdf
h_rec[ismissing.(h_rec)] .= -9999.0
h_rec = Float32.(h_rec)
svd_IceSheetDEM.save_netcdf("output/GP_rec_g$(gr).nc", "data/aerodem/aerodem_rm-filtered_geoid-corr_g$(gr).nc", [h_rec], ["surface"], Dict("surface" => Dict()))
