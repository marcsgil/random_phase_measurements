using LinearAlgebra

function load_fiducial_state(d::Integer, folder::String)
    list = open(joinpath(folder, "zsic_$d.txt")) do file
        list = Float64[]
        for line in eachline(file)
            push!(list, parse(eltype(list), line))
        end
        list
    end

    N = length(list) ÷ 2
    normalize(list[1:N] + im .* list[N+1:2N])
end

function sic_from_fiducial(fiducial)
    d = length(fiducial)
    mub = Array{eltype(fiducial)}(undef, d, d, d)
    buffer = similar(fiducial)
    ω = cis(2π / d)

    for α ∈ axes(mub, 3)
        for l ∈ axes(mub, 2)
            for j ∈ eachindex(buffer, fiducial)
                buffer[j] = ω^(j * α) * fiducial[j]
            end
            circshift!(view(mub, :, l, α), buffer, l)
        end
    end

    mub
end