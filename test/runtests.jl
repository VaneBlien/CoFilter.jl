using Test
using CoFilter

@testset "CoFilter.jl" begin
    include("test_types.jl")
    include("test_relations.jl")
    include("test_similarity.jl")
    include("test_builders.jl")
    include("test_recommenders.jl")
    include("test_system.jl")
    include("test_evaluation.jl")
    include("test_update.jl")
    include("test_distributed.jl")
end