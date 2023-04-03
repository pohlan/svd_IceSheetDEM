using NetCDF

function get_indices(obs, res)
    # load bedmachine mask
    mask            = ncread("data/pism_Greenland_" * res * "m_mcb_jpl_v2023_RAGIS_ctrl.nc", "mask")
    nx, ny          = size(mask)
    not_ocean_mask  = reshape(mask.!= 0. .&& mask .!= 4, nx*ny)  # mask for model grid point that are neither ocean nor outside Greenland
    I_not_ocean     = findall(not_ocean_mask)                    # non-ocean indices
    mask_ice_nested = reshape(mask,nx*ny)[I_not_ocean] .== 2

    # get indices where there is data and ice, with respect to non-ocean-mask
    R         = reshape(obs, length(obs))[I_not_ocean]
    I_marg    = findall(R .!= -9999 .&& R .!= 0.0 .&& mask_ice_nested)
    # get indices of the interior where there is ice and but no data, with respect to non-ocean-mask
    bool_intr = (obs.==-9999 .|| obs .== 0.0)
    for iy = 1:ny
        ix = 1
        while obs[ix,iy] .== -9999
            bool_intr[ix,iy] = false
            ix += 1
            if ix > nx
                break
            end
        end
        ix = size(obs,1)
        while obs[ix,iy] .== -9999
            bool_intr[ix,iy] = false
            ix -= 1
            if ix < 1
                break
            end
        end
    end
    I_intr = findall(reshape(bool_intr,nx*ny,1)[I_not_ocean] .&& mask_ice_nested) ####### mask = ice, data = no interior of ice sheet

    return I_not_ocean, I_marg, I_intr
end
