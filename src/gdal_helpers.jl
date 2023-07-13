import ArchGDAL as AG
using Downloads, Cascadia, Gumbo, HTTP, DataFrames, NetCDF
using DataStructures: OrderedDict

shortread(file) = AG.read(AG.getband(AG.read(file),1))


"""
    get_options(; grid, cut_shp="", x_min=-678650, y_min=-3371600, x_max=905350, y_max=- 635600)
Returns a vector of Strings that can be used as an input for gdalwarp
"""
function get_options(;grid, cut_shp = "", srcnodata="", dstnodata="0", x_min = - 678650,
                                                                      y_min = -3371600,
                                                                      x_max =   905350,
                                                                      y_max = - 635600)
    options = ["-overwrite",
               "-t_srs", "EPSG:3413",
               "-r", "average",
               "-co", "FORMAT=NC4",
               "-co", "COMPRESS=DEFLATE",
               "-co", "ZLEVEL=2",
               "-te", "$x_min", "$y_min", "$x_max", "$y_max",  # y_max and y_min flipped otherwise GDAL is reversing the Y-dimension
               "-tr", "$grid", "$grid",
               ]
    if !isempty(cut_shp)
        append!(options, ["-cutline", cut_shp])
    end
    if !isempty(srcnodata)
        append!(options, ["-srcnodata", srcnodata])
    end
    if !isempty(dstnodata)
        append!(options, ["-dstnodata", dstnodata])
    end
    return options
end

"""
    gdalwarp(path; grid, kwargs...)

taken from https://discourse.julialang.org/t/help-request-using-archgdal-gdalwarp-for-resampling-a-raster-dataset/47039

# Example
```
out = gdalwarp("data/input.nc"; grid=1200, dest="data/output.nc")
```
"""
function gdalwarp(path::String; grid::Real, cut_shp="", srcnodata="", kwargs...)
    ds = AG.read(path) do source
        AG.gdalwarp([source], get_options(;grid, cut_shp, srcnodata); kwargs...) do warped
           band = AG.getband(warped, 1)
           AG.read(band)
       end
    end
    return ds
end
function gdalwarp(path::Vector{String}; grid::Real, cut_shp="", srcnodata="", kwargs...)
    source = AG.read.(path)
    ds = AG.gdalwarp(source, get_options(;grid, srcnodata)) do warped1
        if !isempty(cut_shp)
            AG.gdalwarp([warped1], get_options(;grid, cut_shp, srcnodata); kwargs...) do warped2
                band = AG.getband(warped2, 1)
                AG.read(band)
            end
        else
            band = AG.getband(warped1, 1)
            AG.read(band)
        end
    end
    return ds
end

"""
    save_netcdf(A::Matrix; dest::String, sample_path::String)

- A:            Matrix of field to be saved
- dest:         destination where netcdf should be saved
- sample_path:  path of netcdf file that properties should be copied from
"""
function save_netcdf(A::Matrix; dest::String, sample_path::String, )
    sample_dataset = AG.read(sample_path)
    AG.create(
        dest,
        driver = AG.getdriver(sample_dataset),
        width  = AG.width(sample_dataset),
        height = AG.height(sample_dataset),
        nbands = 1,
        dtype  = Float32,
        options = ["-co", "COMPRESS=DEFLATE"] # reduces the file size
    ) do raster
        AG.write!(raster, A, 1)
        AG.setgeotransform!(raster, AG.getgeotransform(sample_dataset))
        AG.setproj!(raster, AG.getproj(sample_dataset))
    end
end
function save_netcdf(layers::Vector{Matrix}, layernames::Vector{String}, dest, template_path)
    # save to netcdf file
    template_dataset = AG.read(template_path)
        AG.create(
            dest,
            driver = AG.getdriver(template_dataset),
            width  = AG.width(template_dataset),
            height = AG.height(template_dataset),
            nbands = length(layers),
            dtype  = Float32,
            options = ["-co", "COMPRESS=DEFLATE", "-co", "ZLEVEL=6"] # reduces the file size
        ) do raster
            for (i, vals) in enumerate(layers)
                AG.write!(raster, vals, i)
            end
            AG.setgeotransform!(raster, AG.getgeotransform(template_dataset))
            AG.setproj!(raster, AG.getproj(template_dataset))
        end

    # rename layers in bash (don't know how to do it in ArchGDAL directly)
    for (i, ln) in enumerate(layernames)
        run(`ncrename -v $("Band"*string(i)),$ln $dest`)
    end
    return
end

function create_bedmachine_grid(grid; bedmachine_path, template_path)
    # check if bedmachine is there already, otherwise download
    original = bedmachine_path*"BedMachineGreenland-v5.nc"
    if !isfile(original)
        println("Downloading bedmachine..")
        url_bedm = "https://n5eil01u.ecs.nsidc.org/ICEBRIDGE/IDBMG4.005/1993.01.01/BedMachineGreenland-v5.nc"
        Downloads.download(url_bedm, original)
    end

    println("Using gdalwarp to project bedmachine on model grid..")
    # gdalwarp
    layernames = ["mask", "geoid", "bed"]
    fieldvals  = Matrix[]
    for fn in layernames
        new = gdalwarp("NETCDF:"*original*":"*fn;  grid)
        push!(fieldvals, new)
    end

    # save as netcdf
    dest = bedmachine_path * "bedmachine_g$(grid).nc"
    save_netcdf(fieldvals, layernames, dest, template_path)

    return
end

"""
copied from https://gist.github.com/scls19fr/9ea2fd021d5dd9a97271da317bff6533
"""
function get_table_from_html(input::AbstractString)
    r = HTTP.request("GET", input)
    strr=String(r.body)
    h = Gumbo.parsehtml(strr)
    qs = eachmatch(Selector("table"), h.root)
    tables = []
    for helm_table in qs
        column_names = String[]
        d_table = OrderedDict{String, Vector{String}}()
        for (i, row) in enumerate(eachmatch(Selector("tr"), helm_table))
            if (i == 1)
                for (j, colh) in enumerate(eachmatch(Selector("th"), row))
                    colh_text = strip(nodeText(colh))
                    while (colh_text in column_names)  # column header must be unique
                        colh_text = colh_text * "_2"
                    end
                    push!(column_names, colh_text)
                end
            else
                if (i == 2)
                    for colname in column_names
                        d_table[colname] = Vector{String}()
                    end
                end
                for (j, col) in enumerate(eachmatch(Selector("td"), row))
                    col_text = strip(nodeText(col))
                    colname = column_names[j]
                    push!(d_table[colname], col_text)
                end
            end
        end
        df = DataFrame(d_table)
        if length(qs) == 1
            return df
        end
        push!(tables, df)
    end
    return tables
end

function create_aerodem(aerodem_path, imbie_shp_path, bedmachine_path)
    grid      = 150
    raw_path  = aerodem_path*"raw/"
    mkpath(raw_path)

    if isempty(raw_path)
        # Download
        println("Downloading aerodem tif files, this may take a few minutes...")
        url_DEMs          = "https://www.nodc.noaa.gov/archive/arc0088/0145405/1.1/data/0-data/G150AERODEM/DEM/"
        url_reliablt_mask = "https://www.nodc.noaa.gov/archive/arc0088/0145405/1.1/data/0-data/G150AERODEM/ReliabilityMask/"
        for url in [url_DEMs, url_reliablt_mask]
            df = get_table_from_html(url)
            tif_files = df.Name[endswith.(df.Name, ".tif") .&& .!occursin.("carey", df.Name)]
            missing_files = tif_files[.!isfile.(tif_files)]
            Downloads.download.(url .* missing_files, raw_path .* missing_files)
        end
    end

    aerodem_files = glob(raw_path * "aerodem_*.tif")
    rm_files     = glob(raw_path * "rm_*.tif")

    # gdalwarp
    println("Using gdalwarp to merge the aerodem mosaics into one DEM, taking a while..")
    cut_shp = imbie_shp_path*"gris-outline-imbie-1980_updated.shp"
    merged_aero_dest = aerodem_path*"merged_aerodem_g$grid.nc"
    merged_rm_dest   = aerodem_path*"merged_rm_g$grid.nc"
    aero     = gdalwarp(aerodem_files; grid, cut_shp, dest=merged_aero_dest)
    rel_mask = gdalwarp(     rm_files; grid, cut_shp, dest=merged_rm_dest)

    # filter for observations where reliability value is low
    aero_rm_filt = copy(aero)
    aero_rm_filt[rel_mask .< 40] .= 0.0  # only keep values with reliability of at least xx
    # apply geoid correction
    geoid_g150 = bedmachine_path * "geoid_g150.nc"
    geoid      = gdalwarp("NETCDF:"*bedmachine_path*"BedMachineGreenland-v5.nc:geoid";  grid, dest = geoid_g150)

    idx                     = findall(aero_rm_filt .!= 0.0)
    aero_rm_geoid_corr      = zeros(size(aero_rm_filt))
    aero_rm_geoid_corr[idx] = aero_rm_filt[idx] - geoid[idx]
    aero_rm_geoid_corr[aero_rm_geoid_corr .< 0.0] .= 0.0

    # save as netcdf
    sample_path = merged_aero_dest
    dest        = aerodem_path*"aerodem_rm-filtered_geoid-corr_g$(grid).nc"
    save_netcdf(aero_rm_geoid_corr; dest, sample_path)
    return
end

function create_imbie_mask(grid; imbie_path, sample_path)
    # a bit ugly and cumbersome, due to the unability to resolve two issues:
    # 1) cut_shp doesn't work wih too large grids and if src file doesn't correspond to grid size
    # 2) the ArchGDAL gdalwarp function doesn't take Matrices as an input (or at least I haven't figured out how)
    #    need to give a filename as argument
    println("Creating imbie mask netcdf..")
    sample = shortread(sample_path)
    ones_m = ones(size(sample))
    fname_ones = "temp1.nc"
    fname_mask = "temp2.nc"
    save_netcdf(ones_m; dest=fname_ones, sample_path)
    cut_shp = imbie_path * "gris-outline-imbie-1980_updated.shp"
    gdalwarp(fname_ones; grid=150, cut_shp, dest=fname_mask)
    gdalwarp(fname_mask; grid, dest=imbie_path*"imbie_mask_g$(grid).nc")
    run(`rm $fname_ones`)
    run(`rm $fname_mask`)
    return
end
