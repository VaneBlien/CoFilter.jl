using Documenter
using CoFilter

makedocs(
    sitename = "CoFilter.jl",
    modules = [CoFilter],
    pages = [
        "Home" => "index.md",
        "API" => "api.md",
        "Architecture" => "architecture.md",
        "Examples" => "examples.md"
    ]
)
