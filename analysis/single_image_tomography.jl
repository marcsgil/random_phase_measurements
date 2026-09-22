using CairoMakie, FFTW, HDF5, LinearAlgebra, QuantumMeasurements, ProgressMeter, Statistics

result_directory = "results/new/big"
order_directory = joinpath(result_directory, "up_to_order_2")
sigma_index = 1
phase_index = 3

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

remove_background(image, background) = image > background + 1 ? image - background - 1 : zero(image-background)
zero2nan(image) = image > 0 ? image : NaN

function center_crop(image, crop_size)
    row = (size(image, 1) - crop_size) ÷ 2 + 1
    column = (size(image, 2) - crop_size) ÷ 2 + 1
    @view image[row:(row+crop_size-1), column:(column+crop_size-1)]
end

coefficients = h5open(joinpath(order_directory, "modes.h5"), "r") do file
    read(file["coefficients"])
end

phase_factors, basis = h5open(joinpath(result_directory, "prepared.h5"), "r") do file
    order_name = basename(order_directory)
    read(file["phase_factors"]), read(file["orders/$order_name/basis"])
end

images = h5open(joinpath(order_directory, "data.h5"), "r") do file
    read(file["images_phase_fourier"])
end;


basis ./= sqrt.(sum(abs2, basis, dims=(1, 2)))
phase_factor = phase_factors[:, :, phase_index, sigma_index]
phase_fourier_basis = fourier_transform(basis .* phase_factor)

measurement_vectors = (conj.(vector) for vector in eachslice(phase_fourier_basis, dims=(1, 2)))
measurement_matrix = assemble_measurement_matrix(measurement_vectors)
method = MaximumLikelihood()

mkpath("plots")

size(images)

fidelities = Array{Float64}(undef, size(coefficients, 2))

p = Progress(length(fidelities))

Threads.@threads for mode_index in 1:10
    coefficient = coefficients[:, mode_index]
    image = remove_background.(
        images[:, :, mode_index, phase_index, sigma_index],
        background_phase_fourier[:, :, sigma_index],
    )

    theoretical_outcomes = get_probabilities(
        measurement_matrix,
        traceless_vectorization(coefficient),
    )
    theoretical_image = reshape(theoretical_outcomes, size(image))


    rho = estimate_state(vec(image), measurement_matrix, method)[1]
    psi = project2pure(rho)

    fidelities[mode_index] = fidelity(psi, coefficient)

    predicted_image = reshape(get_probabilities(measurement_matrix, traceless_vectorization(psi)), size(image))

    plot_images = (theoretical_image, zero2nan.(image), predicted_image)
    fig_titles = ("Theory", "Experiment", "Prediction")

    if mode_index < 10
        with_theme(theme_latexfonts()) do 
            figure = Figure(size=(1000, 400))

            for n ∈ 1:3
                ax = Axis(figure[1, n], title=fig_titles[n], aspect=1)
                heatmap!(ax, plot_images[n])
                hidedecorations!(ax)
            end
            Label(figure[0, :], "Fidelity: $(round(Int, 100*fidelities[mode_index])) %", fontsize = 16, font = :bold)
            save("plots/temp_$mode_index.png", figure)
        end
    end
    next!(p)
end

print("$(mean(fidelities)) ± $(std(fidelities))")
hist(fidelities, bins=0.7:0.002:1)
