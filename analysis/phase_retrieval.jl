using CairoMakie, FFTW, HDF5, LinearAlgebra, PoissonPhaseRetrieval, ProgressMeter, Statistics

result_directory = "results/new/big"
order_directory = joinpath(result_directory, "up_to_order_4")
sigma_index = 3
phase_index = 2

background_direct, background_fourier, background_phase_fourier = h5open(
    joinpath(result_directory, "background.h5"), "r"
) do file
    read(file["images_direct"]),
    read(file["images_fourier"]),
    read(file["images_phase_fourier"])
end

function fourier_transform(u)
    dims = (1, 2)
    fftshift(fft(ifftshift(u, dims), dims), dims) / sqrt(size(u, 1) * size(u, 2))
end

remove_background(image::T, background) where {T} = image > background + 1 ? image - background - one(T) : zero(image-background - one(T))
zero2nan(image) = image > 0 ? image : NaN

function center_crop(image, crop_size)
    row = (size(image, 1) - crop_size) ÷ 2 + 1
    column = (size(image, 2) - crop_size) ÷ 2 + 1
    @view image[row:(row+crop_size-1), column:(column+crop_size-1)]
end

coefficients = h5open(joinpath(order_directory, "modes.h5"), "r") do file
    read(file["coefficients"])
end

phase_factor, basis = h5open(joinpath(result_directory, "prepared.h5"), "r") do file
    order_name = basename(order_directory)
    file["phase_factors"][:, :, phase_index, sigma_index], read(file["orders/$order_name/basis"])
end

images = h5open(joinpath(order_directory, "data.h5"), "r") do file
    file["images_phase_fourier"][:, :, :, phase_index, sigma_index]
end;


basis ./= sqrt.(sum(abs2, basis, dims=(1, 2)))
phase_fourier_basis = fourier_transform(basis .* phase_factor)
reshapen_phase_fourier_basis = reshape(phase_fourier_basis, :, size(phase_fourier_basis, 3))
##
mkpath("plots")

indices = 1:10

fidelities = Array{Float64}(undef, length(indices))
p = Progress(length(indices))

for mode_index in indices
    coefficient = coefficients[:, mode_index]
    # image = remove_background.(
    #     images[:, :, mode_index],
    #     background_phase_fourier[:, :, sigma_index],
    # )
    image = images[:, :, mode_index]

    y = vec(image)
    b = vec(background_phase_fourier[:, :, sigma_index])
    x0 = optimal_initialization(reshapen_phase_fourier_basis, y, b)
    ψ, loss = poisson_phase_retrieval(reshapen_phase_fourier_basis, x0, y, b, 200, Val(true))

    # println("Computed Loss: $loss")
    # println("----------------")
    # println("Loss at true state: $(loss[end])")
    # println("Loss at solution state: $(compute_loss(ψ, reshapen_phase_fourier_basis, y, b))")

    normalize!(ψ)
    fidelities[mode_index] = abs2(coefficient ⋅ ψ)

    # predicted_image = reshape(get_probabilities(measurement_matrix, traceless_vectorization(psi)), size(image))

    # plot_images = (theoretical_image, zero2nan.(image), predicted_image)
    # fig_titles = ("Theory", "Experiment", "Prediction")

    # if mode_index < 10
    #     with_theme(theme_latexfonts()) do
    #         figure = Figure(size=(1000, 400))

    #         for n ∈ 1:3
    #             ax = Axis(figure[1, n], title=fig_titles[n], aspect=1)
    #             heatmap!(ax, plot_images[n])
    #             hidedecorations!(ax)
    #         end
    #         Label(figure[0, :], "Fidelity: $(round(Int, 100*fidelities[mode_index])) %", fontsize=16, font=:bold)
    #         save("plots/temp_$mode_index.png", figure)
    #     end
    # end
    next!(p)
end

println("$(median(fidelities)) ± $(std(fidelities))")
# hist(fidelities, bins=0.7:0.002:1)

fidelities
##
worst_idx = argmax(fidelities)
θ_worst = traceless_vectorization(coefficients[:, worst_idx])
F_worst = fisher(measurement_matrix, θ_worst)


tr(inv(F_worst))