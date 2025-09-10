module LiMpsSymbolicMemory

using Symbolics
using LinearAlgebra
using Statistics
using Random

export LiMpsEngine, create_memory_entity, store_motif_memory, retrieve_contextual_memories,
       weave_memory_tapestry, compute_memory_coherence, generate_symbolic_narrative,
       create_memory_graph, analyze_memory_patterns, export_limps_data

struct MemoryEntity
    id::String
    type::String
    content::Dict{String, Any}
    symbolic_expression::Any
    weight::Float64
    context::Vector{String}
    relationships::Vector{String}
    timestamp::Float64
    coherence_score::Float64
    narrative_importance::Float64
end

struct MemoryRelationship
    source_id::String
    target_id::String
    relationship_type::String
    strength::Float64
    symbolic_bridge::Any
    context_overlap::Vector{String}
    temporal_proximity::Float64
end

struct LiMpsEngine
    memory_entities::Dict{String, MemoryEntity}
    relationships::Vector{MemoryRelationship}
    symbolic_variables::Dict{Symbol, Any}
    coherence_threshold::Float64
    narrative_weaving_factor::Float64
    memory_decay_rate::Float64
    context_window_size::Int
    max_memory_entities::Int
end

const HASH_SALT = UInt(0x9e3779b97f4a7c15)
stable_hash(s::AbstractString) = Int(mod(hash(s, HASH_SALT), 10_000))

function create_memory_entity(id::String, type::String, content::Dict{String, Any},
                              symbolic_expr::Any, weight::Float64, context::Vector{String})
    coherence_score = calculate_initial_coherence(content, context)
    narrative_importance = calculate_narrative_importance(weight, context)
    return MemoryEntity(
        id, type, content, symbolic_expr, weight, context, String[],
        time(), coherence_score, narrative_importance
    )
end

function calculate_initial_coherence(content::Dict{String, Any}, context::Vector{String})
    content_complexity = length(content) / 10.0
    context_richness = length(context) / 5.0
    symbolic_depth = 0.3 # this engine always attaches a symbolic expression
    coherence = min(1.0, content_complexity + context_richness + symbolic_depth)
    return coherence
end

function calculate_narrative_importance(weight::Float64, context::Vector{String})
    base_importance = weight
    context_multiplier = 1.0 + (length(context) * 0.1)
    if "isolation" in context
        context_multiplier *= 1.2
    end
    if "memory" in context
        context_multiplier *= 1.15
    end
    if "identity" in context
        context_multiplier *= 1.25
    end
    importance = min(1.0, base_importance * context_multiplier)
    return importance
end

function enforce_memory_cap!(engine::LiMpsEngine)
    if length(engine.memory_entities) <= engine.max_memory_entities
        return
    end
    min_id = nothing
    min_score = (Inf, Inf)
    for (id, entity) in engine.memory_entities
        score = (entity.narrative_importance, entity.timestamp)
        if score < min_score
            min_score = score
            min_id = id
        end
    end
    if min_id !== nothing
        delete!(engine.memory_entities, min_id)
        filter!(rel -> rel.source_id != min_id && rel.target_id != min_id, engine.relationships)
    end
end

function store_motif_memory(engine::LiMpsEngine, motif_data::Dict{String, Any})
    motif_id = motif_data["id"]
    motif_type = motif_data["type"]
    properties = motif_data["properties"]
    weight = motif_data["weight"]
    context = motif_data["context"]

    symbolic_expr = create_motif_symbolic_expression(motif_id, properties, context)
    memory_entity = create_memory_entity(
        motif_id, motif_type, properties, symbolic_expr, weight, context
    )

    # Insert first, then enforce cap
    engine.memory_entities[motif_id] = memory_entity
    enforce_memory_cap!(engine)

    relationships = find_memory_relationships(engine, memory_entity)
    append!(engine.relationships, relationships)
    update_entity_relationships(engine, memory_entity, relationships)

    return memory_entity
end

function create_motif_symbolic_expression(motif_id::String, properties::Dict{String, Any},
                                          context::Vector{String})
    @variables m c p t
    expr = 0.0
    expr += (stable_hash(motif_id) % 100) / 100.0 * m
    for (i, ctx) in enumerate(context)
        expr += (stable_hash(ctx) % 100) / 100.0 * c * (i / max(1, length(context)))
    end
    for (key, value) in properties
        if value isa Number
            expr += value * p
        elseif value isa AbstractString
            expr += (stable_hash(value) % 100) / 100.0 * p
        end
    end
    expr += (time() % 1000) / 1000.0 * t
    return expr
end

function find_memory_relationships(engine::LiMpsEngine, new_entity::MemoryEntity)
    relationships = MemoryRelationship[]
    for (id, existing_entity) in engine.memory_entities
        if id != new_entity.id
            context_overlap = intersect(new_entity.context, existing_entity.context)
            if !isempty(context_overlap)
                strength = calculate_relationship_strength(new_entity, existing_entity, context_overlap)
                symbolic_bridge = create_symbolic_bridge(new_entity, existing_entity)
                temporal_proximity = abs(new_entity.timestamp - existing_entity.timestamp) / 3600.0
                temporal_proximity = exp(-temporal_proximity)
                relationship_type = determine_relationship_type(new_entity, existing_entity, context_overlap)
                relationship = MemoryRelationship(
                    new_entity.id, existing_entity.id, relationship_type,
                    strength, symbolic_bridge, context_overlap, temporal_proximity
                )
                push!(relationships, relationship)
            end
        end
    end
    return relationships
end

function calculate_relationship_strength(entity1::MemoryEntity, entity2::MemoryEntity,
                                         context_overlap::Vector{String})
    overlap_ratio = length(context_overlap) / max(1, min(length(entity1.context), length(entity2.context)))
    weight_similarity = max(0.0, 1.0 - abs(entity1.weight - entity2.weight))
    type_compatibility = entity1.type == entity2.type ? 1.0 : 0.5
    context_importance = sum([get_context_importance(ctx) for ctx in context_overlap])
    strength = min(1.0, overlap_ratio * 0.4 + weight_similarity * 0.3 +
                   type_compatibility * 0.2 + context_importance * 0.1)
    return strength
end

function get_context_importance(context::String)
    importance_weights = Dict{String, Float64}(
        "isolation" => 0.9,
        "identity" => 0.9,
        "memory" => 0.8,
        "snake" => 0.8,
        "strand" => 0.7,
        "communication" => 0.7,
        "technology" => 0.6,
        "war" => 0.5,
        "nature" => 0.4
    )
    return get(importance_weights, context, 0.5)
end

function create_symbolic_bridge(entity1::MemoryEntity, entity2::MemoryEntity)
    @variables b s t
    bridge_expr = 0.0
    bridge_expr += (entity1.weight + entity2.weight) / 2.0 * b
    if haskey(entity1.content, "symbolic_expression") && haskey(entity2.content, "symbolic_expression")
        bridge_expr += 0.5 * s
    end
    time_diff = abs(entity1.timestamp - entity2.timestamp)
    bridge_expr += exp(-time_diff / 3600.0) * t
    return bridge_expr
end

function determine_relationship_type(entity1::MemoryEntity, entity2::MemoryEntity,
                                     context_overlap::Vector{String})
    if entity1.type == entity2.type
        return "homogeneous"
    elseif "isolation" in context_overlap
        return "isolated_connection"
    elseif "memory" in context_overlap
        return "memory_link"
    elseif "identity" in context_overlap
        return "identity_mirror"
    elseif "snake" in context_overlap
        return "symbolic_coil"
    elseif "strand" in context_overlap
        return "network_connection"
    else
        return "contextual"
    end
end

function update_entity_relationships(engine::LiMpsEngine, new_entity::MemoryEntity,
                                     relationships::Vector{MemoryRelationship})
    for rel in relationships
        if haskey(engine.memory_entities, rel.source_id)
            push!(engine.memory_entities[rel.source_id].relationships, rel.target_id)
        end
        if haskey(engine.memory_entities, rel.target_id)
            push!(engine.memory_entities[rel.target_id].relationships, rel.source_id)
        end
    end
end

function retrieve_contextual_memories(engine::LiMpsEngine, context::Vector{String}; limit::Int = 10)
    relevance_scores = Dict{String, Float64}()
    for (id, entity) in engine.memory_entities
        context_overlap = intersect(context, entity.context)
        context_score = length(context_overlap) / max(1, max(length(context), length(entity.context)))
        recency_bonus = exp(-(time() - entity.timestamp) / 3600.0)
        importance_bonus = entity.narrative_importance
        coherence_bonus = entity.coherence_score
        relevance_score = context_score * 0.4 + recency_bonus * 0.2 +
                          importance_bonus * 0.2 + coherence_bonus * 0.2
        relevance_scores[id] = relevance_score
    end
    sorted_entities = sort(collect(engine.memory_entities),
                           by = x -> relevance_scores[x[1]], rev = true)
    return [entity for (id, entity) in sorted_entities[1:min(limit, length(sorted_entities))]]
end

function weave_memory_tapestry(engine::LiMpsEngine, focus_context::Vector{String})
    relevant_memories = retrieve_contextual_memories(engine, focus_context, limit = 20)
    @variables tapestry narrative coherence time_flow
    tapestry_expr = 0.0
    for (i, memory) in enumerate(relevant_memories)
        memory_contribution = memory.weight * memory.narrative_importance * memory.coherence_score
        temporal_position = (time() - memory.timestamp) / 3600.0
        temporal_factor = exp(-temporal_position / 24.0)
        context_alignment = length(intersect(focus_context, memory.context)) /
                            max(1, max(length(focus_context), length(memory.context)))
        tapestry_expr += memory_contribution * temporal_factor * context_alignment * tapestry
    end
    coherence_score = compute_memory_coherence(engine, relevant_memories)
    tapestry_expr += coherence_score * coherence
    is_coherent = coherence_score >= engine.coherence_threshold
    return Dict{String, Any}(
        "symbolic_tapestry" => tapestry_expr,
        "relevant_memories" => length(relevant_memories),
        "coherence_score" => coherence_score,
        "meets_threshold" => is_coherent,
        "narrative_complexity" => calculate_narrative_complexity(relevant_memories),
        "temporal_span" => calculate_temporal_span(relevant_memories)
    )
end

function compute_memory_coherence(engine::LiMpsEngine, memories::Vector{MemoryEntity})
    if length(memories) < 2
        return 1.0
    end
    scores = Float64[]
    for i in 1:length(memories), j in (i+1):length(memories)
        rel = find_relationship(engine, memories[i].id, memories[j].id)
        if rel !== nothing
            push!(scores, rel.strength * rel.temporal_proximity)
        end
    end
    mean_score = isempty(scores) ? 0.0 : mean(scores)
    return mean_score
end

function find_relationship(engine::LiMpsEngine, id1::String, id2::String)
    for rel in engine.relationships
        if (rel.source_id == id1 && rel.target_id == id2) ||
           (rel.source_id == id2 && rel.target_id == id1)
            return rel
        end
    end
    return nothing
end

function create_temporal_flow_expression(memories::Vector{MemoryEntity})
    @variables flow time_axis
    if isempty(memories)
        return 0.0
    end
    sorted_memories = sort(memories, by = m -> m.timestamp)
    flow_expr = 0.0
    for i in 1:(length(sorted_memories) - 1)
        time_diff = sorted_memories[i+1].timestamp - sorted_memories[i].timestamp
        flow_expr += exp(-time_diff / 3600.0) * flow
    end
    return flow_expr * time_axis
end

function calculate_narrative_complexity(memories::Vector{MemoryEntity})
    if isempty(memories)
        return 0.0
    end
    all_contexts = Set{String}()
    for memory in memories
        union!(all_contexts, memory.context)
    end
    context_diversity = length(all_contexts) / 9.0
    memory_density = length(memories) / 20.0
    complexity = min(1.0, context_diversity * 0.6 + memory_density * 0.4)
    return complexity
end

function calculate_temporal_span(memories::Vector{MemoryEntity})
    if length(memories) < 2
        return 0.0
    end
    timestamps = [m.timestamp for m in memories]
    span = maximum(timestamps) - minimum(timestamps)
    span_hours = span / 3600.0
    return min(1.0, span_hours / 168.0)
end

function generate_symbolic_narrative(engine::LiMpsEngine, focus_context::Vector{String})
    tapestry = weave_memory_tapestry(engine, focus_context)
    memories = retrieve_contextual_memories(engine, focus_context, limit = 15)
    narrative = Dict{String, Any}(
        "tapestry" => tapestry,
        "memories" => [
            Dict{String, Any}(
                "id" => m.id,
                "type" => m.type,
                "weight" => m.weight,
                "context" => m.context,
                "narrative_importance" => m.narrative_importance,
                "coherence_score" => m.coherence_score
            ) for m in memories
        ],
        "relationships" => extract_narrative_relationships(engine, memories),
        "symbolic_themes" => extract_symbolic_themes(memories),
        "temporal_flow" => create_temporal_narrative(memories)
    )
    return narrative
end

function extract_narrative_relationships(engine::LiMpsEngine, memories::Vector{MemoryEntity})
    relationships = []
    memory_ids = Set([m.id for m in memories])
    for rel in engine.relationships
        if rel.source_id in memory_ids && rel.target_id in memory_ids
            push!(relationships, Dict{String, Any}(
                "source" => rel.source_id,
                "target" => rel.target_id,
                "type" => rel.relationship_type,
                "strength" => rel.strength,
                "context_overlap" => rel.context_overlap
            ))
        end
    end
    return relationships
end

function extract_symbolic_themes(memories::Vector{MemoryEntity})
    theme_counts = Dict{String, Int}()
    for memory in memories
        for context in memory.context
            theme_counts[context] = get(theme_counts, context, 0) + 1
        end
    end
    sorted_themes = sort(collect(theme_counts), by = x -> x[2], rev = true)
    return [Dict{String, Any}("theme" => theme, "frequency" => count)
            for (theme, count) in sorted_themes[1:min(5, length(sorted_themes))]]
end

function create_temporal_narrative(memories::Vector{MemoryEntity})
    if isempty(memories)
        return Dict{String, Any}("events" => [], "temporal_flow" => "static")
    end
    sorted_memories = sort(memories, by = m -> m.timestamp)
    events = []
    for (i, memory) in enumerate(sorted_memories)
        push!(events, Dict{String, Any}(
            "sequence" => i,
            "id" => memory.id,
            "type" => memory.type,
            "timestamp" => memory.timestamp,
            "context" => memory.context,
            "importance" => memory.narrative_importance
        ))
    end
    flow_pattern = length(events) >= 3 ? analyze_temporal_pattern(events) : "linear"
    return Dict{String, Any}(
        "events" => events,
        "temporal_flow" => flow_pattern,
        "total_events" => length(events),
        "time_span" => events[end]["timestamp"] - events[1]["timestamp"]
    )
end

function analyze_temporal_pattern(events::Vector{Dict{String, Any}})
    if length(events) < 3
        return "linear"
    end
    intervals = Float64[]
    for i in 1:(length(events) - 1)
        interval = events[i+1]["timestamp"] - events[i]["timestamp"]
        push!(intervals, interval)
    end
    if all(intervals .> 0)
        if std(intervals) < mean(intervals) * 0.3
            return "rhythmic"
        elseif intervals[end] > mean(intervals) * 2
            return "accelerating"
        elseif intervals[1] > mean(intervals) * 2
            return "decelerating"
        else
            return "irregular"
        end
    else
        return "simultaneous"
    end
end

function create_memory_graph(engine::LiMpsEngine)
    nodes = []
    edges = []
    for (id, entity) in engine.memory_entities
        push!(nodes, Dict{String, Any}(
            "id" => id,
            "type" => entity.type,
            "weight" => entity.weight,
            "context" => entity.context,
            "importance" => entity.narrative_importance,
            "coherence" => entity.coherence_score
        ))
    end
    for rel in engine.relationships
        push!(edges, Dict{String, Any}(
            "source" => rel.source_id,
            "target" => rel.target_id,
            "type" => rel.relationship_type,
            "strength" => rel.strength,
            "context_overlap" => rel.context_overlap
        ))
    end
    return Dict{String, Any}(
        "nodes" => nodes,
        "edges" => edges,
        "total_nodes" => length(nodes),
        "total_edges" => length(edges),
        "graph_density" => length(edges) / max(1, length(nodes) * (length(nodes) - 1) / 2)
    )
end

function analyze_memory_patterns(engine::LiMpsEngine)
    type_counts = Dict{String, Int}()
    for entity in values(engine.memory_entities)
        type_counts[entity.type] = get(type_counts, entity.type, 0) + 1
    end
    context_counts = Dict{String, Int}()
    for entity in values(engine.memory_entities)
        for context in entity.context
            context_counts[context] = get(context_counts, context, 0) + 1
        end
    end
    rel_type_counts = Dict{String, Int}()
    for rel in engine.relationships
        rel_type_counts[rel.relationship_type] = get(rel_type_counts, rel.relationship_type, 0) + 1
    end
    coherence_scores = [entity.coherence_score for entity in values(engine.memory_entities)]
    importance_scores = [entity.narrative_importance for entity in values(engine.memory_entities)]
    return Dict{String, Any}(
        "type_distribution" => type_counts,
        "context_distribution" => context_counts,
        "relationship_types" => rel_type_counts,
        "coherence_stats" => Dict{String, Float64}(
            "mean" => mean(coherence_scores),
            "std" => std(coherence_scores),
            "min" => minimum(coherence_scores),
            "max" => maximum(coherence_scores)
        ),
        "importance_stats" => Dict{String, Float64}(
            "mean" => mean(importance_scores),
            "std" => std(importance_scores),
            "min" => minimum(importance_scores),
            "max" => maximum(importance_scores)
        ),
        "total_entities" => length(engine.memory_entities),
        "total_relationships" => length(engine.relationships)
    )
end

function export_limps_data(engine::LiMpsEngine)
    entities = []
    for (id, entity) in engine.memory_entities
        push!(entities, Dict{String, Any}(
            "id" => id,
            "type" => entity.type,
            "content" => entity.content,
            "symbolic_expression" => string(entity.symbolic_expression),
            "weight" => entity.weight,
            "context" => entity.context,
            "relationships" => entity.relationships,
            "timestamp" => entity.timestamp,
            "coherence_score" => entity.coherence_score,
            "narrative_importance" => entity.narrative_importance
        ))
    end
    relationships = []
    for rel in engine.relationships
        push!(relationships, Dict{String, Any}(
            "source_id" => rel.source_id,
            "target_id" => rel.target_id,
            "relationship_type" => rel.relationship_type,
            "strength" => rel.strength,
            "symbolic_bridge" => string(rel.symbolic_bridge),
            "context_overlap" => rel.context_overlap,
            "temporal_proximity" => rel.temporal_proximity
        ))
    end
    return Dict{String, Any}(
        "memory_entities" => entities,
        "relationships" => relationships,
        "engine_config" => Dict{String, Any}(
            "coherence_threshold" => engine.coherence_threshold,
            "narrative_weaving_factor" => engine.narrative_weaving_factor,
            "memory_decay_rate" => engine.memory_decay_rate,
            "context_window_size" => engine.context_window_size,
            "max_memory_entities" => engine.max_memory_entities
        ),
        "metadata" => Dict{String, Any}(
            "total_entities" => length(entities),
            "total_relationships" => length(relationships),
            "export_timestamp" => time(),
            "version" => "1.0.0"
        )
    )
end

function LiMpsEngine(; coherence_threshold::Float64 = 0.6,
                      narrative_weaving_factor::Float64 = 0.8,
                      memory_decay_rate::Float64 = 0.1,
                      context_window_size::Int = 10,
                      max_memory_entities::Int = 1000)
    @variables m c p t tapestry narrative coherence time_flow
    symbolic_vars = Dict{Symbol, Any}(
        :m => m, :c => c, :p => p, :t => t,
        :tapestry => tapestry, :narrative => narrative,
        :coherence => coherence, :time_flow => time_flow
    )
    return LiMpsEngine(
        Dict{String, MemoryEntity}(),
        MemoryRelationship[],
        symbolic_vars,
        coherence_threshold,
        narrative_weaving_factor,
        memory_decay_rate,
        context_window_size,
        max_memory_entities
    )
end

end # module
