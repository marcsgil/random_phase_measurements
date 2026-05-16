ENV["JULIA_CONDAPKG_BACKEND"] = "Null"
ENV["JULIA_PYTHONCALL_EXE"] = "@venv"

using PythonCall, QuantumMeasurements, CairoMakie, LinearAlgebra, ProgressMeter, Statistics, HDF5, FFTW, ProgressMeter

scipy = pyimport("scipy")
np = pyimport("numpy")
pyimport("sys").path.append(pwd())
utils = pyimport("utils")
h5py = pyimport("h5py")

function remove_backgorund(img, bg)
    map(x -> x > bg ? x - bg : zero(x), img)
end

function match_fourier_basis_to_measurement(u, A_fourier, t_fourier)
    u_fourier = utils.fourier_transform(u)
    scipy.ndimage.affine_transform(u_fourier, scipy.linalg.inv(A_fourier), -np.matmul(scipy.linalg.inv(A_fourier), t_fourier))
end

function match_fourier_basis_to_measurement(u, A_fourier, t_fourier, phase)
    match_fourier_basis_to_measurement(np.exp(pycomplex(im) * phase) * u, A_fourier, t_fourier)
end

function amplitude_fidelity(image, coeffs, basis)
    mode = PyArray(np.sum(coeffs.reshape(-1, 1, 1) * basis, axis=0))
    amplitude_fidelity(image, abs2.(mode))
end

function amplitude_fidelity(image1, image2)
    amplitude1 = sqrt.(image1 / sum(image1))
    amplitude2 = sqrt.(image2 / sum(image2))
    amplitude1 ⋅ amplitude2
end

function linear_combination!(dest, coefficients, basis)
    dest .= 0
    for (c, u) in zip(coefficients, basis)
        dest .+= c * u
    end
end

function linear_combination(coefficients, basis)
    dest = complex(similar(first(basis)))
    linear_combination!(dest, coefficients, basis)
    dest
end

function transform_basis(basis, A, t)
    mapslices(basis, dims=(1, 2)) do u
        scipy.ndimage.affine_transform(u, inv(A), -inv(A) * t) |> PyArray
    end
end

function fourier_transform(u)
    ifftshift(fft(fftshift(u, (1, 2)), (1, 2)), (1, 2))
end

base_dir = "results/controled_exposure/"

A_direct, t_direct, A_fourier, t_fourier = h5open(joinpath(base_dir, "calibration/calibration.h5")) do f
    reverse(read(f["A_direct"]) |> transpose),
    reverse(read(f["t_direct"])),
    reverse(read(f["A_fourier"]) |> transpose),
    reverse(read(f["t_fourier"]))
end
##
mode_idx = 3
phase_idx = 6
sigma_idx = 3

order = 4

direct_fidelities, fourier_fidelities, phase_fourier_fidelities, sigmas = h5open(joinpath(base_dir, "phases.h5")) do f_phases
    sigmas = read(f_phases["sigmas"])
    phases = read(f_phases["phases"])

    phase_transformation = cis.(phases[:, :, phase_idx, sigma_idx])
    h5open(joinpath(base_dir, "up_to_order_$order/modes.h5")) do f_modes
        h5open(joinpath(base_dir, "up_to_order_$order/data.h5")) do f_data

            images_direct = f_data["images_direct"]
            images_fourier = f_data["images_fourier"]
            images_phase_fourier = f_data["images_phase_fourier"]

            coefficients = read(f_modes["coefficients"])

            basis = read(f_modes["basis"])

            @info "Calculating Basis"

            _direct_basis = read(f_modes["basis"])
            _fourier_basis = fourier_transform(_direct_basis)
            _phase_fourier_basis = fourier_transform(_direct_basis .* phase_transformation)

            direct_basis = transform_basis(_direct_basis, A_direct, t_direct)
            fourier_basis = transform_basis(_fourier_basis, A_fourier, t_fourier)
            phase_fourier_basis = transform_basis(_phase_fourier_basis, A_fourier, t_fourier)

            mode = linear_combination(coefficients[:, mode_idx], eachslice(phase_fourier_basis, dims=3))

            with_theme(theme_latexfonts()) do
                fig = Figure()
                ax1 = Axis(fig[1, 1], aspect=1)
                ax2 = Axis(fig[1, 2], aspect=1)
                plot!(ax1, abs2.(mode))
                plot!(ax2, images_phase_fourier[:, :, mode_idx, phase_idx, sigma_idx])
                display(fig)
            end

            @info "Calculating Amplitude Fidelities"

            direct_fidelities = Vector{Float64}(undef, size(images_direct, 3))
            fourier_fidelities = Vector{Float64}(undef, size(images_fourier, 3))

            Threads.@threads for n ∈ eachindex(direct_fidelities, fourier_fidelities)
                image_theo = linear_combination(view(coefficients, :, n), eachslice(direct_basis, dims=3)) .|> abs2
                direct_fidelities[n] = amplitude_fidelity(image_theo, remove_backgorund(images_direct[:, :, n], 2))

                image_theo = linear_combination(view(coefficients, :, n), eachslice(fourier_basis, dims=3)) .|> abs2
                fourier_fidelities[n] = amplitude_fidelity(image_theo, remove_backgorund(images_fourier[:, :, n], 5))
            end

            phase_fourier_fidelities = Array{Float64}(undef, size(images_phase_fourier, 3), size(images_phase_fourier, 4), size(images_phase_fourier, 5))

            for o ∈ axes(phase_fourier_fidelities, 3)
                for n ∈ axes(phase_fourier_fidelities, 2)
                    _phase_fourier_basis = fourier_transform(_direct_basis .* cis.(phases[:, :, n, o]))
                    phase_fourier_basis = transform_basis(_phase_fourier_basis, A_fourier, t_fourier)
                    Threads.@threads for m ∈ axes(phase_fourier_fidelities, 1)
                        image_theo = linear_combination(view(coefficients, :, m), eachslice(phase_fourier_basis, dims=3)) .|> abs2
                        phase_fourier_fidelities[m, n, o] = amplitude_fidelity(image_theo, remove_backgorund(images_phase_fourier[:, :, m, n, o], 5))
                    end
                end
            end

            direct_fidelities, fourier_fidelities, phase_fourier_fidelities, sigmas
        end
    end
end

mean_direct = mean(direct_fidelities)
std_direct = std(direct_fidelities)

mean_fourier = mean(fourier_fidelities)
std_fourier = std(fourier_fidelities)

mean_phase_fourier = dropdims(mean(phase_fourier_fidelities, dims=(1, 2)), dims=(1, 2))
std_phase_fourier = dropdims(std(phase_fourier_fidelities, dims=(1, 2)), dims=(1, 2))

with_theme(theme_latexfonts()) do
    fig = Figure()
    ax = Axis(fig[1, 1], xlabel=L"\sigma", ylabel="Amplitude Fidelity", title="Diagnosis for order up to $order")

    scatter!(ax, sigmas, mean_phase_fourier, color=:black)
    errorbars!(ax, sigmas, mean_phase_fourier, std_phase_fourier, whiskerwidth=10, color=:black)

    hlines!(ax, mean_direct, linestyle=:dash, color=:blue)
    # band!(ax, sigmas, fill(mean_direct-std_direct, length(sigmas)), fill(mean_direct+std_direct, length(sigmas)), alpha = 0.5, color=:blue)

    hlines!(ax, mean_fourier, linestyle=:dash, color=:red)
    # band!(ax, sigmas, fill(mean_fourier-std_fourier, length(sigmas)), fill(mean_direct+std_direct, length(sigmas)), alpha = 0.5, color=:red)

    ylims!(ax, high=1)

    save(joinpath(base_dir, "plots", "diagnosis_$order.png"), fig)
    fig
end
##

for order in 1:4

    # order = 2

    phase_fourier_fidelities, sigmas = h5open(joinpath(base_dir, "phases.h5")) do f_phases
        sigmas = read(f_phases["sigmas"])
        phases = read(f_phases["phases"])

        h5open(joinpath(base_dir, "up_to_order_$order/modes.h5")) do f_modes
            h5open(joinpath(base_dir, "up_to_order_$order/data.h5")) do f_data

                images_phase_fourier = f_data["images_phase_fourier"]

                coefficients = read(f_modes["coefficients"])

                basis = read(f_modes["basis"])

                _direct_basis = read(f_modes["basis"])


                phase_fourier_fidelities = Array{Float64}(undef, size(images_phase_fourier, 3), size(images_phase_fourier, 4), size(images_phase_fourier, 5))

                method = MaximumLikelihood()

                p = Progress(length(phase_fourier_fidelities))

                for o ∈ axes(phase_fourier_fidelities, 3)
                    for n ∈ axes(phase_fourier_fidelities, 2)
                        _phase_fourier_basis = fourier_transform(_direct_basis .* cis.(phases[:, :, n, o]))
                        phase_fourier_basis = transform_basis(_phase_fourier_basis, A_fourier, t_fourier)
                        normalization = sum(abs2, phase_fourier_basis, dims=(1, 2))
                        phase_fourier_basis ./= sqrt.(normalization)
                        μ = assemble_measurement_matrix(conj.(c) for c in eachslice(phase_fourier_basis, dims=(1, 2)))

                        Threads.@threads for m ∈ axes(phase_fourier_fidelities, 1)
                            ρ = estimate_state(remove_backgorund(images_phase_fourier[:, :, m, n, o], 5), μ, method)[1]
                            phase_fourier_fidelities[m, n, o] = fidelity(project2pure(ρ), normalize(view(coefficients, :, m) .* vec(sqrt.(normalization))))
                            # for _ in 1:5
                            #     try
                            #         outcomes = simulate_outcomes(view(coefficients, :, m), μ, 10^5, atol=1e-3)
                            #         ρ = estimate_state(outcomes, μ, method)[1]
                            #         phase_fourier_fidelities[m, n, o] = fidelity(project2pure(ρ), view(coefficients, :, m))
                            #         break
                            #     catch
                            #         continue
                            #     end
                            # end
                            next!(p)
                        end
                    end
                end

                phase_fourier_fidelities, sigmas
            end
        end
    end


    # phase_fourier_fidelities = h5open("results/controled_exposure/fidelities_gs.h5") do f
    #     read(f["up_to_order_$order"])
    # end


    median_phase_fourier = dropdims(median(phase_fourier_fidelities, dims=(1, 2)), dims=(1, 2))
    iqr_phase_fourier = dropdims(mapslices(phase_fourier_fidelities, dims=(1, 2)) do x
            diff(quantile(vec(x), [0.25, 0.75])) / 2
        end, dims=(1, 2))


    with_theme(theme_latexfonts()) do
        fig = Figure()
        ax = Axis(fig[1, 1], xlabel="Fidelity", ylabel="Counts", title="Tomography for order up to $order")

        density!(ax, reshape(view(phase_fourier_fidelities, :, :, 1), :), color=(:red, 0.3), strokecolor=:red, strokewidth=3, strokearound=true,
            label=L"\sigma=%$(sigmas[1]); \ \mathcal{F} = %$(round(100 * median_phase_fourier[1], digits=1)) \pm  %$(round(100 * iqr_phase_fourier[1], digits=1)) \%")
        density!(ax, reshape(view(phase_fourier_fidelities, :, :, 5), :), color=(:blue, 0.3), strokecolor=:blue, strokewidth=3, strokearound=true,
            label=L"\sigma=%$(sigmas[5]); \ \mathcal{F} = %$(round(100 * median_phase_fourier[5], digits=1)) \pm  %$(round(100 * iqr_phase_fourier[5], digits=1)) \%")

        axislegend(ax, position=:lt)

        save(joinpath(base_dir, "plots", "tomography_$order.png"), fig)
        fig
    end
end
