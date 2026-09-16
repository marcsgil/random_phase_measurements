using CairoMakie
using FFTW
using HDF5
using LinearAlgebra
using QuantumMeasurements

result_directory = "results/test"
order_directory = joinpath(result_directory, "up_to_order_1")
sigma_index = 1
phase_index = 1
background = 10

function fourier_transform(u)
    dims = (1, 2)
    fftshift(fft(ifftshift(u, dims), dims), dims) / sqrt(size(u, 1) * size(u, 2))
end

remove_background(image, background) = max.(image .- background, zero(eltype(image)))

function center_crop(image, crop_size)
    row = (size(image, 1) - crop_size) ÷ 2 + 1
    column = (size(image, 2) - crop_size) ÷ 2 + 1
    @view image[row:row+crop_size-1, column:column+crop_size-1]
end

phases = h5open(joinpath(result_directory, "phases.h5"), "r") do file
    read(file["phases"])
end

coefficients, basis = h5open(joinpath(order_directory, "modes.h5"), "r") do file
    read(file["coefficients"]), read(file["basis"])
end

images = h5open(joinpath(order_directory, "prepared.h5"), "r") do file
    read(file["images_phase_fourier"])
end

basis ./= sqrt.(sum(abs2, basis, dims=(1, 2)))
phase = phases[:, :, phase_index, sigma_index]
phase_fourier_basis = fourier_transform(basis .* cis.(phase))

measurement_vectors = (conj.(vector) for vector in eachslice(phase_fourier_basis, dims=(1, 2)))
measurement_matrix = assemble_measurement_matrix(measurement_vectors)
method = MaximumLikelihood()

mkpath("plots")

for mode_index in axes(coefficients, 2)
    coefficient = coefficients[:, mode_index]
    image = remove_background(
        images[:, :, mode_index, phase_index, sigma_index],
        background,
    )

    theoretical_outcomes = get_probabilities(
        measurement_matrix,
        traceless_vectorization(coefficient),
    )
    theoretical_image = reshape(theoretical_outcomes, size(image))

    figure = Figure(size=(800, 400))
    heatmap!(Axis(figure[1, 1]), center_crop(image, 32))
    heatmap!(Axis(figure[1, 2]), center_crop(theoretical_image, 32))
    save("plots/temp.png", figure)

    rho = estimate_state(vec(image), measurement_matrix, method)[1]
    psi = project2pure(rho)
    println(fidelity(psi, coefficient))
end
