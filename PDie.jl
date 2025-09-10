#!/usr/bin/env julia

# Dianne PolyServe + QVNM + CPLearn + Preview + Query + DP (compact)

using HTTP, JSON3, LinearAlgebra, Statistics, Random



##############################

# Util

##############################

epsf() = 1e-12f0

function l2norm!(v::Vector{Float32})

    n = sqrt(sum(x*x for x in v))

    if n > 1e-12f0

        @inbounds for i in eachindex(v); v[i] /= n; end

    end

    v

end



##############################

# (A) Costa–Hero: ID/Entropy

##############################

module CostaHeroID

using NearestNeighbors, Statistics, StatsBase, LinearAlgebra, Random

export id_entropy_global, id_entropy_local, knn_distances



function knn_distances(X::AbstractMatrix{<:Real}; k::Int=10)

    d,N = size(X)

    tree = KDTree(Matrix{Float64}(X))

    D,I = knn(tree, Matrix{Float64}(X), k+1, true) # includes self

    idx = [vec(I[i][2:end]) for i in 1:N]

    dst = [vec(D[i][2:end]) for i in 1:N]

    return idx,dst

end



function L_gamma_subset(S::Vector{Int}, idx_all, dst_all; γ::Float64=1.0)

    present = fill(false, length(idx_all))

    @inbounds for s in S; present[s] = true; end

    total = 0.0

    @inbounds for i in S

        neigh = idx_all[i]; dists = dst_all[i]

        for (j, dij) in zip(neigh, dists)

            if present[j]; total += dij^γ; end

        end

    end

    return total / 2.0

end



ls_fit(xs, ys) = begin

    x̄ = mean(xs); ȳ = mean(ys)

    num = sum((x - x̄)*(y - ȳ) for (x,y) in zip(xs,ys))

    den = sum((x - x̄)^2 for x in xs) + eps()

    a = num/den

    b = ȳ - a*x̄

    a,b

end



function id_entropy_global(X; k::Int=10, γ::Float64=0.5, α::Float64=0.5,

                           plist::Vector{Int}=Int[], boots::Int=16, rng=Random.GLOBAL_RNG)

    d,N = size(X)

    plist = isempty(plist) ? collect(round.(Int, range(max(32, 2k), N; length=8))) : plist

    idx,dst = knn_distances(X; k=k)

    logp = Float64[]; logL = Float64[]

    for p in plist

        μ = 0.0

        trials = min(boots, max(1, cld(N, p)))

        for _ in 1:trials

            S = sample(rng, 1:N, p; replace=false) |> collect

            μ += L_gamma_subset(S, idx, dst; γ=γ)

        end

        μ /= trials

        push!(logp, log(p)); push!(logL, log(max(μ, eps())))

    end

    a,b = ls_fit(logp, logL)

    m_hat = γ / max(1e-9, (1.0 - a))

    H_hat = b / γ

    m_hat, H_hat, (a=a,b=b,γ=γ,α=α,k=k,plist=plist,boots=boots)

end



function id_entropy_local(X; k::Int=10, r::Int=64, γ::Float64=0.5, α::Float64=0.5, boots::Int=8, rng=Random.GLOBAL_RNG)

    d,N = size(X)

    idx,dst = knn_distances(X; k=max(k, r))

    m̂ = fill(Float64(NaN), N); Ĥ = fill(Float64(NaN), N)

    for i in 1:N

        order = sortperm(dst[i])[1:min(r, length(dst[i]))]

        Sfull = vcat(i, idx[i][order]) |> unique |> collect

        P = length(Sfull)

        if P < max(24, 2k); continue; end

        plist = round.(Int, clamp.(range(ceil(Int, 0.3P), P; length=6), 8, P))

        logp=Float64[]; logL=Float64[]

        for p in plist

            μ=0.0; trials=min(boots, max(1, cld(P, p)))

            for _ in 1:trials

                S = sample(rng, Sfull, p; replace=false) |> collect

                μ += L_gamma_subset(S, idx, dst; γ=γ)

            end

            μ/=trials; push!(logp,log(p)); push!(logL,log(max(μ,eps())))

        end

        a,b = ls_fit(logp, logL)

        m̂[i] = γ / max(1e-9, (1.0 - a))

        Ĥ[i] = b / γ

    end

    m̂, Ĥ, (γ=γ, α=α, k=k, r=r, boots=boots)

end



end # CostaHeroID



##############################

# (B) QVNM build/preview

##############################

module QVNM

using LinearAlgebra, Statistics, Graphs, SparseArrays, SimpleWeightedGraphs



export quantum_fidelity, geodesic_dist, blend_weights



quantum_fidelity(V::AbstractMatrix{<:Real}) = begin

    # V: d×N normalized cols

    S = transpose(V) * V

    S .^ 2

end



function geodesic_dist(nei::Vector{Vector{Int}}, w::Vector{Vector{Float64}})

    N = length(nei)

    g = SimpleWeightedGraph(N)

    for i in 1:N

        for (j,wij) in zip(nei[i], w[i])

            add_edge!(g, i, j, wij)

            add_edge!(g, j, i, wij)

        end

    end

    D = fill(Inf, N, N)

    for s in 1:N

        dists = dijkstra_shortest_paths(g, s).dists

        D[s,:] = dists

    end

    D

end



function blend_weights(V::AbstractMatrix{<:Real}, Dg::AbstractMatrix{<:Real},

                       m_hat::AbstractVector{<:Real}, H_hat::AbstractVector{<:Real};

                       k::Int=10, lam_m::Float64=0.3, lam_h::Float64=0.3)

    N = size(V,2)

    sortD = mapslices(sort, Dg; dims=2)[:,1:min(k+1,N)]

    σ = [median(sortD[i,2:end]) for i in 1:N]

    m̄ = mean(m_hat[isfinite.(m_hat)]); H̄ = mean(H_hat[isfinite.(H_hat)])

    for i in 1:N

        σ[i] *= exp(lam_h*((H_hat[i]-H̄)) - lam_m*((m_hat[i]-m̄)))

    end

    # normalize V cols

    Vn = Matrix{Float64}(V)

    for j in 1:N

        n = sqrt(sum(Vn[:,j].^2)); Vn[:,j] ./= (n>1e-12 ? n : 1.0)

    end

    F = quantum_fidelity(Vn)

    S = @. exp(-(Dg^2) / (σ*σ'))

    W = S .* F

    for i in 1:N; W[i,i] = 0.0; end

    Matrix{Float32}(W)

end



end # QVNM



##############################

# (C) QVNM Preview

##############################

module QVNMPreview

using LinearAlgebra, Statistics, JSON3

export preview_from_W



function histo(v::AbstractVector{<:Real}, bins::Int)

    lo,hi = extrema(v)

    if !isfinite(lo) || !isfinite(hi) || hi ≤ lo

        return Dict("bins"=>[], "edges"=>[], "min"=>lo, "max"=>hi, "mean"=>mean(v), "std"=>std(v))

    end

    edges = collect(range(lo, hi; length=bins+1))

    counts = zeros(Int, bins)

    for x in v

        if !isfinite(x); continue; end

        b = clamp(searchsortedlast(edges, x), 1, bins)

        counts[b] += 1

    end

    Dict("bins"=>counts, "edges"=>edges, "min"=>lo, "max"=>hi, "mean"=>mean(v), "std"=>std(v))

end



function preview_from_W(W::AbstractMatrix{<:Real},

                        m_hat::AbstractVector{<:Real},

                        H_hat::AbstractVector{<:Real};

                        r::Int=2, k_eval::Int=10, bins::Int=20)

    N = size(W,1)

    d = vec(sum(W; dims=2))

    deg = Dict("min"=>minimum(d), "max"=>maximum(d), "mean"=>mean(d), "std"=>std(d))

    # normalized kernel S

    ϵ = 1e-12

    Dm12 = 1.0 ./ sqrt.(d .+ ϵ)

    S = (Dm12 .* W) .* transpose(Dm12)

    # eigens (dense fallback)

    ev = eigen(Symmetric(Matrix(S)))

    evals = reverse(ev.values)

    evecs = reverse(ev.vectors, dims=2)

    if length(evals) > k_eval

        evals = evals[1:k_eval]; evecs = evecs[:,1:k_eval]

    end

    gap = length(evals) ≥ 2 ? (evals[1] - evals[2]) : NaN

    rd = min(r, size(evecs,2))

    coords = rd > 0 ? evecs[:,1:rd] : zeros(Float64, N, 0)

    Dict(

        "n"=>N, "edges"=>Int(sum(W .> 0.0) ÷ 2), "degree"=>deg,

        "spectrum"=>Dict("evals"=>evals, "gap"=>gap),

        "eigenmaps"=>Dict("r"=>rd, "coords"=>vec(coords')),

        "histograms"=>Dict("m_hat"=>histo(m_hat, bins), "H_hat"=>histo(H_hat, bins))

    )

end



end # QVNMPreview



##############################

# (D) CPLearn codebook + projector

##############################

module CPLearnCodes

using LinearAlgebra, Statistics, Random

export CodeProj, make_codebook, project_codes, code_affinity, code_hist



struct CodeProj

    W :: Matrix{Float32} # f×c

    τ :: Float32

end



function make_codebook(f::Int, c::Int; seed::Int=2214)

    rng = Random.MersenneTwister(seed)

    W = rand(rng, (-1f0, +1f0), f, c)

    for j in 1:c

        nj = norm(@view W[:,j]); if nj>0; @views W[:,j] .= (sqrt(f)/nj).*W[:,j]; end

    end

    W

end



function row_softmax!(S::Matrix{Float32})

    N,c = size(S)

    @inbounds for i in 1:N

        m = maximum(@view S[i,:]); s=0.0f0

        for j in 1:c; S[i,j] = exp(S[i,j]-m); s+=S[i,j]; end

        invs = 1.0f0/max(s,1e-12f0)

        for j in 1:c; S[i,j]*=invs; end

    end

    S

end



function project_codes(H::AbstractMatrix{<:Real}, cp::CodeProj)

    N,f = size(H); fW,c = size(cp.W); @assert f==fW

    logits = Matrix{Float32}(H) * cp.W

    logits ./= max(cp.τ, 1e-6f0)

    P = row_softmax!(logits)

    hard = Vector{Int}(undef, N)

    @inbounds for i in 1:N

        best=1; bv=P[i,1]

        for j in 2:c

            v=P[i,j]; if v>bv; best=j; bv=v; end

        end

        hard[i]=best

    end

    P,hard

end



function code_affinity(P::AbstractMatrix{<:Real}, hard_idx::Vector{Int}; hard::Bool=false)

    N,c = size(P)

    if hard

        A = zeros(Float32, N,N)

        @inbounds for i in 1:N

            ci=hard_idx[i]; A[i,i]=1.0f0

            for j in i+1:N

                v = (ci==hard_idx[j]) ? 1.0f0 : 0.0f0

                A[i,j]=v; A[j,i]=v

            end

        end

        return A

    else

        A = Matrix{Float32}(P) * Matrix{Float32}(P')

        mx = maximum(A); if mx>0; A./=mx; end

        return A

    end

end



function code_hist(hard_idx::Vector{Int}, c::Int)

    cnt = zeros(Int, c); @inbounds for x in hard_idx; if 1≤x≤c; cnt[x]+=1; end; end

    cnt

end



end # CPLearnCodes



# active code projector

global CPL_CP = CPLearnCodes.CodeProj(CPLearnCodes.make_codebook(256, 4096), 0.07f0)



function qvnm_with_codes(W::AbstractMatrix{<:Real}, H::AbstractMatrix{<:Real},

                         cp::CPLearnCodes.CodeProj; lambda_code::Float64=0.25, hard::Bool=false)

    P, hard_idx = CPLearnCodes.project_codes(H, cp)

    A = CPLearnCodes.code_affinity(P, hard_idx; hard=hard)

    Wf = (1.0 - lambda_code) .* Matrix{Float32}(W) .+ lambda_code .* Matrix{Float32}(A)

    hist = CPLearnCodes.code_hist(hard_idx, size(cp.W,2))

    Wf, P, hard_idx, hist

end



##############################

# (E) DP (compact) — optional

##############################

module DPCollapse

export dp_diffusion_frictionless, dp_sigma_eq_smallbeta, dp_beta_critical

function dp_diffusion_frictionless(m,R0; G,ħ)

    (G*ħ*m^2) / (3*sqrt(pi)*R0^3)

end

function dp_sigma_eq_smallbeta(m,R0,ω,β; ħ)

    (4/(ω*β)) * (1 + (3*ħ^2*β)/(4*m*R0^2))

end

function dp_beta_critical(m,R0,ω; ħ)

    denom = (ħ*ω - 3*ħ^2/(m*R0^2))

    4/denom

end

end



##############################

# Handlers

##############################

router = HTTP.Router()



# Health

HTTP.register!(router, "GET", "/health", r->HTTP.Response(200, "ok"))



# (1) Estimate ID/entropy

using .CostaHeroID

function handle_estimate_id(req::HTTP.Request)

    o = JSON3.read(String(req.body))

    d = Int(o["d"]); N = Int(o["N"])

    V = reshape(Vector{Float32}(o["V"]), d, N) |> Array{Float64}

    k = Int(get(o, "k", 10)); γ = Float64(get(o,"gamma",0.5)); α = Float64(get(o,"alpha",0.5));

    boots = Int(get(o,"boots",8)); mode = String(get(o,"mode","local"))

    if mode=="global"

        m,H,diag = CostaHeroID.id_entropy_global(V; k=k, γ=γ, α=α, boots=boots)

        return HTTP.Response(200, JSON3.write(Dict("mode"=>"global","m_hat"=>m,"H_hat"=>H,"diag"=>Dict(diag))))

    else

        r = Int(get(o,"r",64))

        m,H,diag = CostaHeroID.id_entropy_local(V; k=k, γ=γ, α=α, boots=boots, r=r)

        return HTTP.Response(200, JSON3.write(Dict("mode"=>"local","m_hat"=>m,"H_hat"=>H,"diag"=>Dict(diag))))

    end

end

HTTP.register!(router, "POST", "/qvnm/estimate_id", handle_estimate_id)



# (2) Build QVNM

using .QVNM

function qvnm_build(o)

    # expects: V(d×N col-major), neighbors, weights, m_hat, H_hat, lambda_m, lambda_h

    d = Int(o["d"]); N = Int(o["N"])

    V = reshape(Vector{Float32}(o["V"]), d, N) |> Array{Float64}

    nei = [Vector{Int}(x) for x in o["neighbors"]]

    wts = [Vector{Float64}(x) for x in o["weights"]]

    Dg = QVNM.geodesic_dist(nei, wts)

    m = [Float64(x) for x in o["m_hat"]]; H = [Float64(x) for x in o["H_hat"]]

    lam_m = Float64(get(o,"lambda_m",0.3)); lam_h = Float64(get(o,"lambda_h",0.3))

    W = QVNM.blend_weights(V, Dg, m, H; k=min(10, size(Dg,2)-1), lam_m=lam_m, lam_h=lam_h)

    Dict("W"=>W)

end



function handle_build(req::HTTP.Request)

    o = JSON3.read(String(req.body))

    out = qvnm_build(o)

    HTTP.Response(200, JSON3.write(out))

end

HTTP.register!(router, "POST", "/qvnm/build", handle_build)



# (3) Preview

using .QVNMPreview

function handle_preview(req::HTTP.Request)

    o = JSON3.read(String(req.body))

    mode = String(get(o,"mode","W"))

    r    = Int(get(o,"r",2)); k_eval = Int(get(o,"k_eval",10)); bins = Int(get(o,"bins",20))

    if mode=="build"

        out = qvnm_build(o)

        W = Array{Float64}(out["W"])

        m = [Float64(x) for x in o["m_hat"]]; H = [Float64(x) for x in o["H_hat"]]

        prev = QVNMPreview.preview_from_W(W, m, H; r=r, k_eval=k_eval, bins=bins)

        return HTTP.Response(200, JSON3.write(prev))

    else

        N = Int(o["N"]); Wvec = Vector{Float64}(o["W"])

        W = reshape(permutedims(reshape(Wvec,(N,N))), (N,N))

        m = [Float64(x) for x in o["m_hat"]]; H = [Float64(x) for x in o["H_hat"]]

        prev = QVNMPreview.preview_from_W(W, m, H; r=r, k_eval=k_eval, bins=bins)

        return HTTP.Response(200, JSON3.write(prev))

    end

end

HTTP.register!(router, "POST", "/qvnm/preview", handle_preview)



# (4) Query (diffusion walk with optional prior)

function row_stochastic(W)

    N = size(W,1)

    P = Matrix{Float64}(W); rs = sum(P; dims=2)

    for i in 1:N

        s = rs[i]

        if s>0; P[i,:] ./= s; end

    end

    P

end

function diffusion_walk(P, psi0; alpha=0.85, steps=10, theta=0.0, prior=nothing)

    N = size(P,1)

    ψ = collect(Float64, psi0); s0=sum(abs.(ψ)); ψ ./= (s0>0 ? s0 : 1.0)

    blend = copy(ψ)

    if prior !== nothing

        ρ = collect(Float64, prior); ρ .-= minimum(ρ); sρ=sum(ρ); ρ./=(sρ>0 ? sρ : 1.0/N)

        @. blend = (1-theta)*ψ + theta*ρ

    end

    for _ in 1:max(1,steps)

        ψ = alpha * (P * ψ) .+ (1-alpha)*blend

    end

    s = sum(abs.(ψ)); ψ./=(s>0 ? s : 1.0)

    ψ

end

function topk(x::AbstractVector{<:Real}, k::Int)

    N=length(x); k=min(max(k,1),N)

    idx = partialsortperm(x, rev=true, 1:k)

    [(i, x[i]) for i in idx]

end



function handle_query(req::HTTP.Request)

    o = JSON3.read(String(req.body))

    mode = String(get(o,"mode","W"))

    topK = Int(get(o,"topk",5)); steps=Int(get(o,"steps",10)); alpha=Float64(get(o,"alpha",0.85))

    theta=Float64(get(o,"theta",0.0))

    ids = get(o,"ids", nothing)



    W = nothing

    if mode=="build"

        out = qvnm_build(o); W = Array{Float64}(out["W"])

    else

        N = Int(o["N"]); Wvec = Vector{Float64}(o["W"])

        W = reshape(permutedims(reshape(Wvec,(N,N))), (N,N))

    end



    N = size(W,1); Pmat = row_stochastic(W)

    ψ0 = fill(1.0/N, N)

    if haskey(o,"seed_id") && ids !== nothing && o["seed_id"] !== nothing

        idlist = [String(x) for x in ids]; seed = String(o["seed_id"])

        pos = findfirst(==(seed), idlist)

        if pos !== nothing; ψ0 .= 0.0; ψ0[pos]=1.0; end

    end



    prior = haskey(o,"prior") ? [Float64(x) for x in o["prior"]] : nothing

    ψT = diffusion_walk(Pmat, ψ0; alpha=alpha, steps=steps, theta=theta, prior=prior)

    winners = topk(ψT, topK)

    results = Any[]

    if ids !== nothing

        idlist = [String(x) for x in ids]

        for (i,s) in winners; push!(results, Dict("id"=>idlist[i], "idx"=>i, "score"=>s)); end

    else

        for (i,s) in winners; push!(results, Dict("idx"=>i, "score"=>s)); end

    end

    HTTP.Response(200, JSON3.write(Dict("top"=>results,"meta"=>Dict("N"=>N,"steps"=>steps,"alpha"=>alpha,"theta"=>theta,"mode"=>mode))))

end

HTTP.register!(router, "POST", "/qvnm/query", handle_query)



# (4b) Query trajectory: return ψ for each step (length = steps+1)

function handle_query_traj(req::HTTP.Request)

    o = JSON3.read(String(req.body))

    mode   = String(get(o, "mode", "build"))

    steps  = Int(get(o, "steps", 10))

    alpha  = Float64(get(o, "alpha", 0.85))

    theta  = Float64(get(o, "theta", 0.0))

    ids    = get(o, "ids", nothing)



    # Build/accept W

    W = if mode == "build"

        Array{Float64}(qvnm_build(o)["W"])

    else

        N = Int(o["N"])

        Wvec = Vector{Float64}(o["W"])

        reshape(permutedims(reshape(Wvec, (N, N))), (N, N))

    end

    N = size(W, 1)

    Pmat = row_stochastic(W)



    # Seed ψ0

    ψ0 = fill(1.0 / N, N)

    if haskey(o, "seed_id") && ids !== nothing && o["seed_id"] !== nothing

        idlist = [String(x) for x in ids]; seed = String(o["seed_id"])

        pos = findfirst(==(seed), idlist)

        if pos !== nothing

            ψ0 .= 0.0; ψ0[pos] = 1.0

        end

    end

    prior = haskey(o, "prior") ? [Float64(x) for x in o["prior"]] : nothing



    # Roll out trajectory

    traj = Vector{Vector{Float64}}(undef, steps + 1)

    ψ = copy(ψ0); traj[1] = copy(ψ)

    for t in 1:steps

        ψ = diffusion_walk(Pmat, ψ; alpha = alpha, steps = 1, theta = theta, prior = prior)

        traj[t + 1] = copy(ψ)

    end



    return HTTP.Response(200, JSON3.write(Dict(

        "N" => N, "steps" => steps, "alpha" => alpha, "theta" => theta, "traj" => traj

    )))

end

HTTP.register!(router, "POST", "/qvnm/query_traj", handle_query_traj)




main

# (5) Build+codes

function handle_build_codes(req::HTTP.Request)

    o = JSON3.read(String(req.body))

    λc = Float64(get(o,"lambda_code",0.25)); hard = Bool(get(o,"hard",false))

    out = qvnm_build(o); W = Array{Float32}(out["W"])

    d = Int(o["d"]); N = Int(o["N"])

    V = reshape(Vector{Float32}(o["V"]), d, N) # d×N

    H = tanh.(Matrix{Float32}(V'))            # N×d

    Wf,P,hard_idx,hist = qvnm_with_codes(W, H, CPL_CP; lambda_code=λc, hard=hard)

    HTTP.Response(200, JSON3.write(Dict("W"=>Wf,"codes"=>Dict("hist"=>hist,"hard_idx"=>hard_idx,"f"=>size(CPL_CP.W,1),"c"=>size(CPL_CP.W,2),"tau"=>CPL_CP.τ))))

end

HTTP.register!(router, "POST", "/qvnm/build_codes", handle_build_codes)



# (6) Codes init

function handle_cpl_init(req::HTTP.Request)

    o = JSON3.read(String(req.body))

    f = Int(get(o,"f",256)); c = Int(get(o,"c",4096)); τ = Float32(get(o,"tau",0.07)); seed=Int(get(o,"seed",2214))

    global CPL_CP = CPLearnCodes.CodeProj(CPLearnCodes.make_codebook(f,c;seed=seed), τ)

    HTTP.Response(200, JSON3.write(Dict("ok"=>true,"f"=>f,"c"=>c,"tau"=>τ)))

end

HTTP.register!(router, "POST", "/cpl/init", handle_cpl_init)



# (7) DP summary

using .DPCollapse

function handle_dp_summary(req::HTTP.Request)

    o = JSON3.read(String(req.body))

    m=Float64(get(o,"m",1.0)); R0=Float64(get(o,"R0",1e-9)); ω=Float64(get(o,"omega",1.0)); β=Float64(get(o,"beta",1e-5))

    G=Float64(get(o,"G",6.67430e-11)); ħ=Float64(get(o,"hbar",1.054571817e-34))

    Df = DPCollapse.dp_diffusion_frictionless(m,R0; G=G,ħ=ħ)

    σeq2 = DPCollapse.dp_sigma_eq_smallbeta(m,R0,ω,β; ħ=ħ)

    βc = DPCollapse.dp_beta_critical(m,R0,ω; ħ=ħ)

    HTTP.Response(200, JSON3.write(Dict("frictionless_D"=>Df,"sigma_eq2"=>σeq2,"beta_c"=>βc)))

end

HTTP.register!(router, "POST", "/dp/summary", handle_dp_summary)



# Serve

port = parse(Int, get(ENV,"PORT","9000"))

println("PolyServe+QVNM on :$port")

HTTP.serve(router, ip"0.0.0.0", port)



# Julia deps (install from REPL):
