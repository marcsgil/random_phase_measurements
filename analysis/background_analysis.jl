using CairoMakie
using HDF5

result_directory = "results/test"
order_directory = joinpath(result_directory, "up_to_order_1")

images = h5open(joinpath(order_directory, "data.h5"), "r") do file
    read(file["images_phase_fourier"])
end;

image = images[:,:,1,1,1]


with_theme(theme_latexfonts()) do
    fig = Figure(size=(800,400))
    ax1 = Axis(fig[1,1])
    hist!(ax1,vec(image), bins=0:maximum(image))
    ylims!(ax1, 0, 1e3)
    ax2 = Axis(fig[1,2], aspect=1)
    heatmap!(ax2,image, colorrange=(0,255))
    fig
end
