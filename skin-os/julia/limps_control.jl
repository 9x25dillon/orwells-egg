using HTTP, JSON3

function control(req)
    try
        q = HTTP.URI(req.target).query
        avg = parse(Int, get(q, "avg", "5"))
        inflow = max(1, min(20, Int(round(0.9 * avg + 1))))
        return HTTP.Response(200, JSON3.write(Dict("inflow_hz" => inflow)))
    catch
        return HTTP.Response(200, JSON3.write(Dict("inflow_hz" => 5)))
    end
end

HTTP.serve() do req::HTTP.Request
    if startswith(String(req.target), "/control")
        return control(req)
    else
        return HTTP.Response(200, "ok")
    end
end
# runs on 0.0.0.0:8081 by default in newer HTTP.jl
