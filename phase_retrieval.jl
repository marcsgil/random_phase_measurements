ENV["JULIA_CONDAPKG_BACKEND"] = "Null"
ENV["JULIA_PYTHONCALL_EXE"] = "@venv"

using PythonCall, QuantumMeasurements, CairoMakie, LinearAlgebra, ProgressMeter, Statistics, HDF5, FFTW, ProgressMeter, PoissonPhaseRetrieval

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
mode_idx = 10
phase_idx = 1
sigma_idx = 1

order = 2

sigmas, phase = h5open(joinpath(base_dir, "phases.h5")) do f_phases
    read(f_phases["sigmas"]), f_phases["phases"][:, :, phase_idx, sigma_idx]
end

_direct_basis, coefficients = h5open(joinpath(base_dir, "up_to_order_$order/modes.h5")) do f_modes
    read(f_modes["basis"]), f_modes["coefficients"][:, mode_idx]
end

image = h5open(joinpath(base_dir, "up_to_order_$order/data.h5")) do f_data
    f_data["images_phase_fourier"][:, :, mode_idx, phase_idx, sigma_idx]
end

b = fill(6, length(phase))



_phase_fourier_basis = fourier_transform(_direct_basis .* cis.(phase))
phase_fourier_basis = transform_basis(_phase_fourier_basis, A_fourier, t_fourier)

# for slice ∈ eachslice(phase_fourier_basis, dims=3)
#     normalize!(slice)
# end


μ = assemble_measurement_matrix(conj.(c) for c in eachslice(phase_fourier_basis, dims=(1, 2)))

sensing_operator = reshape(phase_fourier_basis, :, size(phase_fourier_basis, 3))

plot(abs2.(phase_fourier_basis[:, :, 1]))

outcomes = simulate_outcomes(coefficients, μ, 10^3, atol=1e-3)
probs = get_probabilities(μ, traceless_vectorization(coefficients))

# probs = get_probabilities(μ, traceless_vectorization(normalize(ψ)))



plot(reshape(probs, size(phase)))
plot(image)

# x0 = rand(ComplexF32, length(coefficients))

y = reshape(image, :)
# y = outcomes
x0 = optimal_initialization(sensing_operator, y, b)

ψ, losses = poisson_phase_retrieval(sensing_operator, x0, y, b, 50)
normalize!(ψ)

fidelity(ψ, coefficients)

##
fig = Figure()
ax = Axis(fig[1, 1], title="Fidelity: $(round(fidelity(ψ, coefficients), digits=2))")
lines!(ax, losses)
# ylims!(ax, 2e6, 5e6)
fig
##

##
phase_fourier_fidelities, sigmas = h5open(joinpath(base_dir, "phases.h5")) do f_phases
    sigmas = read(f_phases["sigmas"])
    phases = read(f_phases["phases"])

    h5open(joinpath(base_dir, "up_to_order_$order/modes.h5")) do f_modes
        h5open(joinpath(base_dir, "up_to_order_$order/data.h5")) do f_data

            images_phase_fourier = f_data["images_phase_fourier"]

            coefficients = read(f_modes["coefficients"])

            basis = read(f_modes["basis"])



            phase_fourier_fidelities = Array{Float64}(undef, size(images_phase_fourier, 3), size(images_phase_fourier, 4), 1)

            p = Progress(length(phase_fourier_fidelities))

            # x0 = rand(ComplexF32, (order + 1) * (order + 2) ÷ 2)
            b = fill(6, size(images_phase_fourier, 1) * size(images_phase_fourier, 2))

            for o ∈ axes(phase_fourier_fidelities, 3)
                for n ∈ axes(phase_fourier_fidelities, 2)
                    _phase_fourier_basis = fourier_transform(_direct_basis .* cis.(phases[:, :, n, o]))
                    phase_fourier_basis = transform_basis(_phase_fourier_basis, A_fourier, t_fourier)
                    normalization = sum(abs2, phase_fourier_basis, dims=(1, 2))
                    phase_fourier_basis ./= sqrt.(normalization)

                    μ = assemble_measurement_matrix(conj.(c) for c in eachslice(phase_fourier_basis, dims=(1, 2)))
                    phase_fourier_basis = reshape(phase_fourier_basis, :, size(phase_fourier_basis, 3))

                    for m ∈ axes(phase_fourier_fidelities, 1)
                        # y = reshape(images_phase_fourier[:, :, m, n, o], :)
                        # x0 = optimal_initialization(phase_fourier_basis, y, b)
                        # ψ = poisson_phase_retrieval(phase_fourier_basis, x0, y, b, 200)[1]
                        # normalize!(ψ)
                        # phase_fourier_fidelities[m, n, o] = fidelity(ψ, view(coefficients, :, m))
                        for _ in 1:5
                            try
                                outcomes = simulate_outcomes(view(coefficients, :, m), μ, 10^5, atol=1e-3)
                                y = reshape(outcomes, :)
                                x0 = optimal_initialization(phase_fourier_basis, y, b)
                                ψ = poisson_phase_retrieval(phase_fourier_basis, x0, y, b, 200)[1]
                                normalize!(ψ)
                                phase_fourier_fidelities[m, n, o] = fidelity(ψ, view(coefficients, :, m))
                                break
                            catch
                                continue
                            end
                        end
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

phase_fourier_fidelities


median_phase_fourier = dropdims(median(phase_fourier_fidelities, dims=(1, 2)), dims=(1, 2))
iqr_phase_fourier = dropdims(mapslices(phase_fourier_fidelities, dims=(1, 2)) do x
        diff(quantile(vec(x), [0.25, 0.75])) / 2
    end, dims=(1, 2))


with_theme(theme_latexfonts()) do
    fig = Figure()
    ax = Axis(fig[1, 1], xlabel="Fidelity", ylabel="Counts", title="Tomography for order up to $order")

    density!(ax, reshape(view(phase_fourier_fidelities, :, :, 1), :), color=(:red, 0.3), strokecolor=:red, strokewidth=3, strokearound=true,
        label=L"\sigma=%$(sigmas[1]); \ \mathcal{F} = %$(round(100 * median_phase_fourier[1], digits=1)) \pm  %$(round(100 * iqr_phase_fourier[1], digits=1)) \%")
    # density!(ax, reshape(view(phase_fourier_fidelities, :, :, 5), :), color=(:blue, 0.3), strokecolor=:blue, strokewidth=3, strokearound=true,
    #     label=L"\sigma=%$(sigmas[5]); \ \mathcal{F} = %$(round(100 * median_phase_fourier[5], digits=1)) \pm  %$(round(100 * iqr_phase_fourier[5], digits=1)) \%")

    axislegend(ax, position=:lt)

    # save(joinpath(base_dir, "plots", "tomography_$order.png"), fig)
    fig
end