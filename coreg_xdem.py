import xdem
import rioxarray
import geoutils as gu
import numpy as np
import matplotlib.pyplot as plt

# choose target resolution
gr = 1200
# choose resampling method
r_method = "average"

# load data
aero_not_aligned = xdem.DEM("data/aerodem/aerodem_rm-filtered_geoid-corr_g150.nc")
grimp_reference  = xdem.DEM("data/bedmachine/bedmachine_g150_surface.nc")   # NOTE: this is not an automated output of svd_IceSheetDEM
atm_not_aligned  = xdem.DEM("data/ATM/ATM_elevation_geoid_corrected_g150.nc")

# co-registration function
def do_align(reference_dem, dem_to_be_aligned):
    init_dh = reference_dem - dem_to_be_aligned
    # Create a mask of stable terrain, removing outliers outside 3 NMAD
    glacier_outlines = gu.Vector("data/gris-imbie-1980/gris-outline-imbie-1980_updated.shp")
    mask_noglacier   = ~glacier_outlines.create_mask(reference_dem)
    mask_nooutliers  = np.abs(init_dh - np.nanmedian(init_dh)) < 3 * xdem.spatialstats.nmad(init_dh)
    # Create inlier mask
    inlier_mask      = mask_noglacier & mask_nooutliers

    # calculate dh before
    diff_before = reference_dem - dem_to_be_aligned
    # co-register
    nuth_kaab = xdem.coreg.NuthKaab()
    print("Doing Nuth and Kaab fit...")
    nuth_kaab.fit(reference_dem, dem_to_be_aligned, inlier_mask)
    print(nuth_kaab._meta)
    aligned_dem = nuth_kaab.apply(dem_to_be_aligned)
    # calculate dh after
    diff_after = reference_dem - aligned_dem
    return aligned_dem, diff_before, diff_after

# function to save as a netcdf on a different grid
def save_netcdf_at_grid(grid, data, valname, dest_file, r_method):
    dem_gr = data.reproject(dst_res=grid, resampling=r_method) # resampling="bilinear" is default but doesn't work well for ATM
    dem_xa = dem_gr.to_xarray(valname)
    dem_xa.to_netcdf(dest_file)

# aerodem
aero_aligned, aero_diff_before, aero_diff_after = do_align(grimp_reference, aero_not_aligned)
save_netcdf_at_grid(gr, aero_diff_after, "dh", f"data/grimp_minus_aero_g{gr}.nc", r_method)
save_netcdf_at_grid(gr, aero_aligned, "surface", f"data/aerodem/aerodem_g{gr}_aligned_"+r_method+".nc", r_method)

# atm
atm_aligned, atm_diff_before, atm_diff_after = do_align(grimp_reference, atm_not_aligned)
save_netcdf_at_grid(gr, atm_diff_after, "dh", f"data/grimp_minus_atm_g{gr}.nc", r_method)
save_netcdf_at_grid(gr, atm_aligned, "surface", f"data/ATM/ATM_elevation_g{gr}_"+r_method+".nc", r_method)
