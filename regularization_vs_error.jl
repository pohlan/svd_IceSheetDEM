using svd_IceSheetDEM, Glob, NCDatasets, JLD2
import Plots

parsed_args = parse_commandline(ARGS)
model_files = parsed_args["training_data"]
shp_file    = parsed_args["shp_file"]
use_arpack  = parsed_args["use_arpack"]

template_file = model_files[1]
x = NCDataset(template_file)["x"]
y = NCDataset(template_file)["y"]
const gr = Int(x[2] - x[1])
const F  = Float32

bedm_file              = create_bedmachine_grid(gr, template_file)
bedmachine_path        = splitdir(bedm_file)[1]
aerodem_g150, obs_file = create_aerodem(;gr, shp_file, bedmachine_path)
imbie_mask             = create_imbie_mask(;gr, shp_file, sample_path=aerodem_g150)

fig_dir = "output/figures/"
mkpath(fig_dir)
destdir = "output/model_selection/"
mkpath(destdir)

# give λ and r values to loop through
λs        = [1e4, 1e5, 1e6, 5e6, 1e7, 5e7, 1e8, 5e8, 1e9, 5e9]
rs        = [50, 100, 150, 200, 250, 300, 350]

# load datasets, take full SVD (to be truncated with different rs later)
UΣ, I_no_ocean, Data_mean, Σ = svd_IceSheetDEM.prepare_model(model_files, imbie_mask, bedm_file, maximum(rs), F, use_arpack)
x_data, I_obs                = svd_IceSheetDEM.prepare_obs(obs_file, I_no_ocean, Data_mean)
f_eval(λ, r, i_train       ) = svd_IceSheetDEM.solve_optim(UΣ, I_obs[i_train], r, λ, x_data[i_train])

# create sets of training and test data
k = 10
i_train_sets, i_test_sets = svd_IceSheetDEM.k_fold_gaps(I_obs, k)

# loop through λ and r values
dict      = svd_IceSheetDEM.sample_param_space(f_eval, λs, rs, x_data, I_obs, i_train_sets, i_test_sets)
dict["σ"] = Σ
dest      = destdir*"dict_$(k)-fold_cv.jld2"
save(dest, dict)

# plot
for m in keys(filter(p -> .!(p.first ∈ ["r","λ", "σ"]), dict))
    Plots.plot(λs, abs.(dict[m][:,1]), xscale=:log10, xlabel="λ", ylabel=m, label="r="*string(rs[1]), marker=:circle, markerstrokewidth=0)
    for i in 2:length(rs)
        Plots.plot!(λs, abs.(dict[m][:,i]), xscale=:log10, xlabel="λ", ylabel=m, label="r="*string(rs[i]), marker=:circle, markerstrokewidth=0)
    end
    Plots.savefig(fig_dir*m*"_abs_$(k)-fold_cv.png")
end
