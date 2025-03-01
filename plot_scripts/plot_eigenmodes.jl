using JLD2, Plots, LaTeXStrings

fig_dir = "output/SVD_reconstruction/figures"


########################
# plot eigenmodes      #
########################

# define how many modes to plot
i_modes = [1,2,3,100]

# load
d = load("output/data_preprocessing/SVD_components.jld2")
U = d["U"][:,i_modes]

prms = load("output/data_preprocessing/params_gr600.jld2")
I_no_ocean = prms["I_no_ocean"]

nx, ny = 2640, 4560   # hard-coded for 600m

b = zeros(nx*ny, length(i_modes))
b[I_no_ocean, :] .= U
b[b .== 0] .= NaN

Plots.scalefontsizes()
Plots.scalefontsizes(3.0)

colors = cgrad(:RdBu)

# plot eigenmodes
ps = [heatmap(reshape(b[:,i], nx,ny)', clims=(-2e-3, 2e-3), cmap=:RdBu, showaxis=false, top_margin = -20Plots.mm, leftmargin=-30Plots.mm, grid=false, title= latexstring("u_{$(i_modes[i])}"), cbar=false, aspect_ration=true) for i in 1:length(i_modes)]
p_panels = plot(ps..., size=(2100,900), aspect_ratio=1, left_margin=[0Plots.mm 0Plots.cm], right_margin=[0Plots.mm 0Plots.mm], bottom_margin=-9Plots.mm, layout=Plots.grid(1,length(ps)))

# add a colorbar separately (inspired by https://discourse.julialang.org/t/set-custom-colorbar-tick-labels/69806/4)
xx = range(0,1,500)
zz = zero(xx)' .+ xx
p_c = heatmap(xx, xx, zz, ticks=false, ratio=25, legend=false, fc=colors, lims=(0,1), clims=(0,1), framestyle=:box, right_margin=20Plots.mm, top_margin=10Plots.mm, bottom_margin=15Plots.mm)
yticks  = [0.05, 0.5, 0.95]
ticktxt = ["negative", "0", "positive"]
[annotate!(1.5, yi, text(ti, 20, "Computer Modern", :left)) for (yi,ti) in zip(yticks,ticktxt)]

# plot again everything together
layout = @layout [a{0.9w,1.0h} b{0.7h}]
plot(p_panels, p_c; layout, bottom_margin=-40Plots.mm, size=(2100,600), top_margin=10Plots.mm)
Plots.savefig(joinpath(fig_dir, "eigenmodes.png"))



########################
# plot v vectors       #
########################

v_rec = d["v_rec"]

bar(d["v_rec"][1:100], label="", linecolor=:cornflowerblue, color=:cornflowerblue, size=attr.size, margin=attr.margin, xlabel=L"Mode index $i$", ylabel=L"$\mathbf{v}_\mathrm{rec,\,i}$", grid=false)
savefig(joinpath(fig_dir, "v_rec.png"))

# plot!(d["V"][20,:], label="")

# histogram(d["v_rec"], bins=30, linecolor=:cornflowerblue, color=:cornflowerblue, label="", xlabel=L"Values of $\mathbf{v}_\mathrm{rec}$", ylabel="Count", grid=false, size=attr.size, margin=attr.margin)
# savefig("v_rec.png")
