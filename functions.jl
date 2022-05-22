using Dates, DataFrames, CSV, JuMP

function read_input(fp)
    
    data = CSV.File(fp, normalizenames=true) |> DataFrame;
    @info "# of rows in raw data = " size(data)[1]
    
    # Create daterange for Timestamp column
    dr = DateTime(2019):Hour(1):DateTime(2019,12,31,23)
    @info "# of rows in daterange = " size(dr)[1]

    # Set timestamp = daterange for easier processing
    data.Timestamp = dr
    return data
end

# Create a matrix so we can containerize objective coefficients
# per https://jump.dev/JuMP.jl/stable/tutorials/linear/factory_schedule/
function create_cost_mat(upper_lim::Int)
    mat = zeros(2,upper_lim);
    mat[1,:] .= 0.0; # Solar price is free
    mat[2,:] = data.da_prices; # grid prices = from data
    println(mat[:,1:5])
    return mat
end;

function create_cost_mat_storage(upper_lim::Int)
    mat = zeros(3,upper_lim);
    mat[1,:] .= 0.0; # Storage price is free
    mat[2,:] = data.da_prices; # grid prices = from data
    mat[3,:] = data.da_prices; # storage charge prices = from data
    println(mat[:,1:5])
    return mat
end;

function create_cost_mat_both(upper_lim::Int)
    mat = zeros(4,upper_lim);
    mat[1,:] .= 0.0; # Solar dispatch is free
    mat[2,:] .= 0.0; # Storage dispatch is free
    mat[3,:] = data.da_prices; # grid prices = from data
    mat[4,:] = data.da_prices; # storage charge prices = from data
    println(mat[:,1:5])
    return mat
end;

containerize(A::Matrix, P, T) = Containers.DenseAxisArray(A, P, T);

function solar_optimize()
    mat = create_cost_mat(upper_lim)
    size(mat)
    
    model = Model(CPLEX.Optimizer)
    
    P = [:solar, :grid]
    T = collect(1:upper_lim); # Timesteps
    D = repeat([1], upper_lim); # 1 MWh demand for each timestep
    
    cost_coeff = containerize(mat, P, T)
    
    # Decision variable = energy produced per hour, indexed over plants and timesteps
    # Also added non-negativity constraint
    @variable(model, x[P, T] >= 0.0)
    
    # Constraint: supply >= demand
    @constraint(model, [t in T], sum(x[:, t]) >= 1.0)
    
    # Solar generation has to be less than CF
    @constraint(model, [t in T], x[:solar, t] <= data.solar_cf_perkw[t])
    
    # Minimize the total cost of meeting demand
    @objective(
        model,
        Min,
        sum(cost_coeff[p, t] * x[p, t] for p in P, t in T)
    )
    
    optimize!(model)
    
    return model, x
end;

function storage_optimize()
    
    mat = create_cost_mat_storage(upper_lim)

    model = Model(CPLEX.Optimizer)

    T = collect(1:upper_lim); # Timesteps

    # stor_disch is the amount of energy being dispatched from storage between 0, 0.25
    # grid is the energy dispatched from grid, between 0, 1 per timestep
    P = [:stor_disch, :grid, :stor_ch]
    @variable(model, a[P, T] >= 0.0)

    # Binary variables to charge/discharge at a timestep
    B = [:disch, :ch]
    @variable(model, c[B, T], binary=true)

    # DISCHARGING
    @constraint(model, sum(c[:disch, t] for t in T) >= 3000)
    @constraint(model, sum(c[:disch, t] for t in T) <= 6000)

    # For all timesteps, if discharge, then storage discharges <= 0.25
    @constraint(model, [t in T], c[:disch, t] => {
        a[:stor_disch, t] <= 0.25
        }
    )

    # Constraint: supply >= demand for all timesteps
    @constraint(model, [t in T], sum(a[:stor_disch, t] + a[:grid, t]) >= 1.0)

    # CHARGING
    @constraint(model, sum(c[:ch, t] for t in T) >= 3000)
    @constraint(model, sum(c[:ch, t] for t in T) <= 6000)

    # For all timesteps, if discharge, then storage discharges <= 0.25
    @constraint(model, [t in T], c[:disch, t] => {
        a[:stor_ch, t] >= 0.25
        }
    )

    # # For all timesteps, can either charge or discharge
    @constraint(model, [t in T], c[:disch, t] + c[:ch, t] <= 1.0)

    # # For all timesteps, energy capacity <= 10.0 MWh
    S = [:bess_mwh]
    @variable(model, b[S, T])
    @constraint(model, [t in T], sum(b[:bess_mwh, t] + a[:stor_ch, t]) <= 1.0)
    @constraint(model, [t in T], sum(b[:bess_mwh, t] - a[:stor_disch, t]) >= 0.0)
    @constraint(model, sum(b[:bess_mwh, t] for t in T) <= 4380.0)
    
    # Storage charge/discharge equality constraint
    @constraint(model, sum(a[:stor_disch, t] for t in T) - sum(a[:stor_ch, t] for t in T) <= 0.01)
    
    cost_coeff = containerize(mat, P, T)

    # Minimize the total cost of meeting demand
    @objective(
        model,
        Min,
        sum(cost_coeff[p, t] * a[p, t] for p in P, t in T)
    )

    optimize!(model)
    
    return model, a, b, c
end;

function solar_storage()

    mat = create_cost_mat_both(upper_lim)

    model = Model(CPLEX.Optimizer)

    T = collect(1:upper_lim); # Timesteps

    # stor_disch is the amount of energy being dispatched from storage between 0, 0.25
    # grid is the energy dispatched from grid, between 0, 1 per timestep
    P = [:solar, :stor_disch, :grid, :stor_ch]
    @variable(model, a[P, T] >= 0.0)

    # Binary variables to charge/discharge at a timestep
    B = [:disch, :ch]
    @variable(model, c[B, T], binary=true)

    # DISCHARGING
    @constraint(model, sum(c[:disch, t] for t in T) >= 3000)
    @constraint(model, sum(c[:disch, t] for t in T) <= 6000)

    # For all timesteps, if discharge, then storage discharges <= 0.25
    @constraint(model, [t in T], c[:disch, t] => {
        a[:stor_disch, t] <= 0.25
        }
    )

    # Constraint: supply >= demand for all timesteps
    @constraint(model, [t in T], sum(a[:solar, t] + a[:stor_disch, t] + a[:grid, t]) >= 1.0)

    # CHARGING
    @constraint(model, sum(c[:ch, t] for t in T) >= 3000)
    @constraint(model, sum(c[:ch, t] for t in T) <= 6000)

    # For all timesteps, if charge, then storage discharges <= 0.25
    @constraint(model, [t in T], c[:disch, t] => {
        a[:stor_ch, t] >= 0.25
        }
    )
    
    # # For all timesteps, can either charge or discharge
    @constraint(model, [t in T], c[:disch, t] + c[:ch, t] <= 1.0)

    # # For all timesteps, energy capacity <= 10.0 MWh
    S = [:bess_mwh]
    @variable(model, b[S, T])
    @constraint(model, [t in T], sum(b[:bess_mwh, t] + a[:stor_ch, t]) <= 1.0)
    @constraint(model, [t in T], sum(b[:bess_mwh, t] - a[:stor_disch, t]) >= 0.0)
    @constraint(model, sum(b[:bess_mwh, t] for t in T) <= 4380.0)
    
    # Storage charge/discharge equality constraint
    @constraint(model, sum(a[:stor_disch, t] for t in T) - sum(a[:stor_ch, t] for t in T) <= 0.01)

    # Solar generation has to be less than CF
    @constraint(model, [t in T], a[:solar, t] <= data.solar_cf_perkw[t])

    cost_coeff = containerize(mat, P, T)

    # Minimize the total cost of meeting demand
    @objective(
        model,
        Min,
        sum(cost_coeff[p, t] * a[p, t] for p in P, t in T)
    )

    optimize!(model)
    
    return model, a, b, c
end;