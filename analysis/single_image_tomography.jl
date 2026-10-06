using CairoMakie, FFTW, HDF5, LinearAlgebra, QuantumMeasurements, ProgressMeter, Statistics, FourierTools

result_directory = "results/test"
order_directory = joinpath(result_directory, "up_to_order_4")
sigma_index = 4
phase_index = 4

remove_background(image::T, background) where {T} = image > background + 1 ? image - background - one(T) : zero(image-background - one(T))
zero2nan(image) = image > 0 ? image : NaN

coefficients, basis = h5open(joinpath(order_directory, "modes.h5")) do file
    read(file["coefficients"]), read(file["basis"])
end

phase = h5open(joinpath(result_directory, "phases.h5")) do file
    file["phases"][:, :, phase_index, sigma_index]
end

images = h5open(joinpath(order_directory, "data.h5"), "r") do file
    file["images_phase_fourier"][:, :, :, phase_index, sigma_index]
end;

zoom, offset = h5open(joinpath(result_directory, "calibration_data", "calibration.h5")) do file
    read(file["zoom"]), reverse(read(file["shift"]))
end

phase_fourier_basis = czt(basis .* cis.(phase), (zoom, zoom, 1), (1, 2))
phase_fourier_basis ./= sqrt.(sum(abs2, phase_fourier_basis, dims=(1, 2)))
phase_fourier_basis = shift(phase_fourier_basis, (-offset[1], -offset[2], 0))

measurement_vectors = (conj.(vector) for vector in eachslice(phase_fourier_basis, dims=(1, 2)))
measurement_matrix = assemble_measurement_matrix(measurement_vectors)

method = MaximumLikelihood()

mkpath("plots")
##
indices = 1:10

fidelities = Array{Float64}(undef, length(indices))
p = Progress(length(indices))

Threads.@threads for mode_index in indices
    coefficient = coefficients[:, mode_index]
    image = remove_background.(
        images[:, :, mode_index],
        4,
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

    plot_images = (theoretical_image, image, predicted_image)
    fig_titles = ("Theory", "Experiment", "Prediction")

    if mode_index < 10
        with_theme(theme_latexfonts()) do
            figure = Figure(size=(1000, 400))

            for n ∈ 1:3
                ax = Axis(figure[1, n], title=fig_titles[n], aspect=1)
                heatmap!(ax, plot_images[n])
                hidedecorations!(ax)
            end
            Label(figure[0, :], "Fidelity: $(round(Int, 100*fidelities[mode_index])) %", fontsize=16, font=:bold)
            save("plots/temp_$mode_index.png", figure)
        end
    end
    next!(p)
end

print("$(median(fidelities)) ± $(std(fidelities))")
##
worst_idx = argmax(fidelities)
θ_worst = traceless_vectorization(coefficients[:, worst_idx])
F_worst = fisher(measurement_matrix, θ_worst)

tr(inv(F_worst))