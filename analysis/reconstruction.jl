using QuantumMeasurements, PoissonPhaseRetrieval, HDF5, FourierTools, ProgressMeter, LinearAlgebra, CairoMakie

function remove_background(value, background)
    diff = value - background
    value > background ? diff : zero(diff)
end

function load_basis(dir, phase, zoom, offset)
    basis = h5open(joinpath(dir, "modes.h5")) do file
        read(file["basis"])
    end

    phase_fourier_basis = czt(basis .* cis.(phase), (zoom, zoom, 1), (1, 2))
    phase_fourier_basis ./= sqrt.(sum(abs2, phase_fourier_basis, dims=(1, 2)))
    shift(phase_fourier_basis, (offset[1], offset[2], 0))
end

struct Tomography{T}
    measurement_matrix::T
    function Tomography(basis)
        measurement_vectors = (conj.(vector) for vector in eachslice(basis, dims=(1, 2)))
        measurement_matrix = assemble_measurement_matrix(measurement_vectors)
        new{typeof(measurement_matrix)}(measurement_matrix)
    end
end

struct PhaseRetrieval{T}
    sensing_operator::T
    function PhaseRetrieval(basis)
        sensing_operator = reshape(basis, :, size(basis, 3))
        new{typeof(sensing_operator)}(sensing_operator)
    end
end

function reconstruct(image, background, method::Tomography)
    ρ = estimate_state(remove_background.(image, background), method.measurement_matrix, MaximumLikelihood())[1]
    project2pure(ρ)
end

generate_background(x::AbstractArray, y) = x
generate_background(x::Number, y) = fill(x * one(eltype(y)), length(y))

function reconstruct(image, background, method::PhaseRetrieval, iterations=100)
    y = vec(image)
    b = generate_background(background, y)
    x0 = optimal_initialization(method.sensing_operator, y, b)
    ψ = poisson_phase_retrieval(method.sensing_operator, x0, y, b, iterations)[1]
    normalize!(ψ)
end

function predict(ψ, basis)
    dropdims(abs2.(sum(reshape(ψ, 1, 1, :) .* basis, dims=3)), dims=3)
end
##
result_directory = "results/test"
order_directory = joinpath(result_directory, "up_to_order_4")

zoom, offset = h5open(joinpath(result_directory, "calibration_data", "calibration.h5")) do file
    read(file["zoom"]), reverse(read(file["shift"]))
end

coefficients = h5open(joinpath(order_directory, "modes.h5")) do file
    read(file["coefficients"])
end

phases = h5open(joinpath(result_directory, "phases.h5")) do file
    read(file["phases"])
end

fidelities = h5open(joinpath(order_directory, "data.h5"), "r") do file
    images_dataset = file["images_phase_fourier"]
    fidelities = Array{Float64}(undef, size(images_dataset)[3:end])

    h5open(joinpath(order_directory, "fidelities.h5"), "cw") do f
        if haskey(f, "fidelities")
            throw(ArgumentError("Cannot create dataset, it already exists"))
        end
    end

    p = Progress(length(fidelities))

    for sigma_index ∈ axes(fidelities, 3), phase_index ∈ axes(fidelities, 2)
        images = images_dataset[:, :, :, phase_index, sigma_index]
        phase = @view phases[:, :, phase_index, sigma_index]
        basis = load_basis(order_directory, phase, zoom, offset)
        method = PhaseRetrieval(basis)

        for mode_index ∈ axes(fidelities, 1)
            coeffs = @view coefficients[:, mode_index]
            image = @view images[:, :, mode_index]
            ψ = reconstruct(image, 3, method)
            fidelities[mode_index, phase_index, sigma_index] = abs2(coeffs ⋅ ψ)

            if mode_index ≤ 5 && phase_index ≤ 4
                plot_directory = joinpath(order_directory, "plots", "sigma_index_$sigma_index", "phase_index_$phase_index")
                mkpath(plot_directory)

                theoretical_image = predict(coeffs, basis)
                predicted_image = predict(ψ, basis)

                plot_images = (theoretical_image, image, predicted_image)
                fig_titles = ("Theory", "Experiment", "Prediction")

                with_theme(theme_latexfonts()) do
                    figure = Figure(size=(1000, 400))
                    for n ∈ 1:3
                        ax = Axis(figure[1, n], title=fig_titles[n], aspect=1)
                        heatmap!(ax, plot_images[n])
                        hidedecorations!(ax)
                    end
                    method_name = string(nameof(typeof(method)))
                    Label(figure[0, :], "Fidelity: $(round(Int, 100*fidelities[mode_index, phase_index, sigma_index])) %", fontsize=16, font=:bold)
                    save(joinpath(plot_directory, "$method_name$mode_index.png"), figure)
                end
            end

            next!(p)
        end
    end

    h5open(joinpath(order_directory, "fidelities.h5"), "w") do f
        f["fidelities"] = fidelities
    end

    fidelities
end
fidelities
##
dropdims(mean(fidelities, dims=1), dims=1)
##
