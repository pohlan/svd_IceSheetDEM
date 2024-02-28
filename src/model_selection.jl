function loocv_gaps(x, y, r_gap, n_gaps, I_no_ocean, I_obs)
    get_x(i,nx) = i % nx == 0 ? nx : i % nx
    get_y(i,nx) = cld(i,nx)
    xobs = x[get_x.(I_no_ocean[I_obs],length(x))]
    yobs = y[get_y.(I_no_ocean[I_obs],length(x))]

    # create circular gaps centered around random points
    i_centers = unique(rand(eachindex(I_obs), n_gaps))
    i_train_sets = []
    i_test_sets  = []
    @showprogress for ic in i_centers
        xc, yc = xobs[ic], yobs[ic]
        dist = pairwise(euclidean, [xc yc], [xobs yobs], dims=1)[:]
        i_gap = findall(dist .< r_gap)
        i_train = Vector(1:length(I_obs))
        deleteat!(i_train, i_gap)
        @assert length(i_gap) + length(i_train) == length(I_obs)
        push!(i_train_sets, i_train)
        push!(i_test_sets, i_gap)
    end
    return i_train_sets, i_test_sets
end

function k_fold_gaps(I_obs, k)
    n = length(I_obs)
    n_per_gap = cld(n, k)
    i_train_sets = []
    i_test_sets  = []
    for i0 in 1:n_per_gap:n
        i_test = i0:min(i0+n_per_gap-1, n)
        i_train = Vector(1:length(I_obs))
        deleteat!(i_train, i_test)
        @assert length(i_test) + length(i_train) == length(I_obs)
        push!(i_train_sets, i_train)
        push!(i_test_sets, i_test)
    end
    return i_train_sets, i_test_sets
end

function initiate_dict(λs, rs, methods_name)
    d = Dict{String,Array}(n => zeros(length(λs), length(rs)) for n in methods_name)
    d["λ"] = λs
    d["r"] = rs
    return d
end

function evaluate_params(f_eval, λ, r, x_data, I_obs, i_train_sets, i_test_sets)
    merrs = []
    for (i_train, i_test) in zip(i_train_sets, i_test_sets)
        x_rec = f_eval(λ, r, i_train)
        difs = x_rec[I_obs[i_test]] .- x_data[i_test]
        push!(merrs, difs)
    end
    all_errs  = vcat(merrs...)
    return all_errs
end

function sample_param_space(f_eval, λs, rs, inputs...)
    # loop through r and λ values and return given metrics
    methods_name = ["median", "mean", "nmad", "std", "L2norm"]
    methods_fct  = [median, mean, mad, std, norm]
    dict = initiate_dict(λs, rs, methods_name)
    for (iλ,λ) in enumerate(λs)
        for (ir,r) in enumerate(rs)
            logλ = round(log(10, λ),digits=1)
            println("r = $r, logλ = $logλ")
            errs  = evaluate_params(f_eval, λ, r, inputs...)
            for (mn, mf) in zip(methods_name, methods_fct)
                dict[mn][iλ,ir] = mf(errs)
            end
        end
    end
    return dict
end
