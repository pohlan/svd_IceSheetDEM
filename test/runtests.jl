using svd_IceSheetDEM
using Test, LinearAlgebra, NetCDF, NCDatasets

cd(@__DIR__)  # set working directory to where file is located

const gr = 4000

template_file       = "testdata/testtemplate_g$(gr).nc"
shp_file            = "testdata/testshape.shp"

rm("data/", recursive=true, force=true)
rm("output/", recursive=true, force=true)

###################################################
# testing the data downloading and pre-processing #
###################################################

# bedmachine
bedmachine_file_gr = create_bedmachine_grid(gr, template_file)
bedmachine_path    = splitdir(bedmachine_file_gr)[1]
# aerodem
aero_150_file, aero_gr_file = create_aerodem(;gr, shp_file, bedmachine_path, kw="1981")    # only download the DEMs from 1981 while testing to save some space
# imbie mask
imbie_mask_file = create_imbie_mask(;gr, shp_file, sample_path=aero_150_file)
# atm
atm_file = create_atm_grid(gr, bedmachine_file_gr, "1994.05.21/")


missmax(x) = maximum(x[.!ismissing.(x)])
missmin(x) = minimum(x[.!ismissing.(x)])
missum(x)  = sum(x[.!ismissing.(x)])

@testset "bedmachine" begin
    ds = NCDataset(bedmachine_file_gr)
    @test all(["mask","geoid","bed","surface","thickness"] .∈ (keys(ds),))
    mask = ds["mask"][:]
    @test missum(mask.==1) .== 39235 && missum(mask.==2) .== 95036 && missum(mask.==3) .== 92 && missum(mask.==4) .== 7052
    @test missmax(ds["geoid"][:]) == 64 && missmin(ds["geoid"][:]) == 6 && sum(ismissing.(ds["geoid"][:])) == 8208
    @test missmax(ds["bed"][:]) == 3106.2961f0 && missmin(ds["bed"][:]) == -5521.8154f0
    close(ds)
end

@testset "aerodem" begin
    ds150                 = NCDataset(aero_150_file)["surface"][:]
    dsgr                  = NCDataset(aero_gr_file )["Band1"][:]
    @test missmax(ds150)  ≈ 3685.5481
    @test sum(.!ismissing.(dsgr))  == 7022    && missmax(dsgr)   ≈ 3252.3076
end

@testset "imbie mask" begin
    imb = NCDataset(imbie_mask_file)["Band1"]
    @test missum(imb .== 1) == sum(.!ismissing.(imb)) == 9496
end

@testset "atm" begin
    atm = ncread(atm_file, "surface")
    @test sum(atm .> 0) == 633
    @test maximum(atm) == 2385.68f0
end

###############################
# testing the problem solving #
###############################

F           = Float32
λ           = 1e5
r           = 10^3
rec_file    = do_reconstruction(F, λ, r, gr, imbie_mask_file, bedmachine_file_gr, [template_file], aero_gr_file)
rec_bm_file = create_reconstructed_bedmachine(rec_file, bedmachine_file_gr)

@testset "solve least square fit" begin
    rec         = NCDataset(rec_file)["surface"][:,:]
    @test missmax(rec) ≈ 3041.7896f0 && rec[353:356, 326] ≈ Float32[2008.46, 2018.27, 1980.02, 1889.64] && missum(rec .> 0) == 2542
    bm          = NCDataset(rec_bm_file)
    @test all(["mask","bed","surface","thickness","polar_stereographic","x","y"] .∈ (keys(bm),))
    mask = bm["mask"][:]
    @test sum(mask.==1) == 139044 && sum(mask.==2) == 2388 && sum(mask.==3) == 55
    @test isapprox(missmax(bm["surface"][:]), 3106, atol=1)
    close(bm)
end

####################################
# testing the uncertainty analysis #
####################################
