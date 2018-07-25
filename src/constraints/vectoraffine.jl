#=
    Vector valued constraints
=#
constrdict(m::LinQuadOptimizer, ::VLCI{MOI.Nonnegatives}) = cmap(m).nonnegatives
constrdict(m::LinQuadOptimizer, ::VLCI{MOI.Nonpositives}) = cmap(m).nonpositives
constrdict(m::LinQuadOptimizer, ::VLCI{MOI.Zeros})        = cmap(m).zeros

function MOI.addconstraint!(m::LinQuadOptimizer, func::VecLin, set::S) where S <: VecLinSets
    @assert MOI.dimension(set) == length(func.constants)

    nrows = get_number_linear_constraints(m)
    addlinearconstraint!(m, func, backend_type(m,set))
    nrows2 = get_number_linear_constraints(m)

    m.last_constraint_reference += 1
    ref = VLCI{S}(m.last_constraint_reference)

    dict = constrdict(m, ref)
    dict[ref] = collect(nrows+1:nrows2)
    for i in 1:MOI.dimension(set)
        push!(m.constraint_primal_solution, NaN)
        push!(m.constraint_dual_solution, NaN)
        push!(m.constraint_constant, func.constants[i])
    end
    ref
end

function addlinearconstraint!(m::LinQuadOptimizer, func::VecLin, sense::Cchar)
    outputindex   = [term.output_index for term in func.terms]
    columns       = [getcol(m, term.scalar_term.variable_index) for term in func.terms]
    coefficients  = [term.scalar_term.coefficient for term in func.terms]
    # sort into row order
    pidx = sortperm(outputindex)
    permute!(columns, pidx)
    permute!(coefficients, pidx)

    # check that there is at least a RHS for each row
    @assert maximum(outputindex) <= length(func.constants)
    # loop through to get starting position of each row
    row_pointers = Vector{Int}(undef, length(func.constants))
    row_pointers[1] = 1
    row = 1
    for i in 2:length(pidx)
        if outputindex[pidx[i]] != outputindex[pidx[i-1]]
            row += 1
            row_pointers[row] = i
        end
    end
    A = CSRMatrix{Float64}(row_pointers, columns, coefficients)
    add_linear_constraints!(m, A, fill(sense, length(func.constants)), -func.constants)
end

MOI.canmodify(m::LinQuadOptimizer, ::Type{VLCI{S}}, ::Type{MOI.VectorConstantChange{Float64}}) where S <: VecLinSets = true
function MOI.modify!(m::LinQuadOptimizer, ref::VLCI{<: VecLinSets}, chg::MOI.VectorConstantChange{Float64})
    @assert length(chg.new_constant) == length(m[ref])
    for (r, v) in zip(m[ref], chg.new_constant)
        change_rhs_coefficient!(m, r, -v)
        m.constraint_constant[r] = v
    end
end

MOI.canmodify(m::LinQuadOptimizer, ::Type{VLCI{S}}, ::Type{MOI.MultirowChange{Float64}}) where S <: VecLinSets = true
function MOI.modify!(m::LinQuadOptimizer, ref::VLCI{<: VecLinSets}, chg::MOI.MultirowChange{Float64})
    col = m.variable_mapping[chg.variable]
    for (row, coef) in chg.new_coefficients
        change_matrix_coefficient!(m, row, col, coef)
    end
end

function matching_sparsity_pattern(f1::VecLin, f2::VecLin)
    indices(f1.terms) == indices(f2.terms) || return false
    indices(f1.constants) == indices(f2.constants) || return false
    for i in eachindex(f1.terms)
        f1.terms[i].output_index == f2.terms[i].output_index || return false
        f1.terms[i].scalar_term.variable_index == f2.terms[i].scalar_term.variable_index || return false
    end
    return true
end

function replace_with_matching_sparsity!(m::LinQuadOptimizer, previous::VecLin, replacement::VecLin, constraint_indices)
    rows = [constraint_indices[t.output_index] for t in previous.terms]
    cols = [m.variable_mapping[t.scalar_term.variable_index] for t in previous.terms]
    coefs = [t.scalar_term.coefficient for t in replacement.terms]
    change_matrix_coefficients!(m, rows, cols, coefs)
end

function replace_with_different_sparsity!(m::LinQuadOptimizer, previous::VecLin, replacement::VecLin, constraint_indices)
    # First, zero out the old constraint function terms
    rows = [constraint_indices[t.output_index] for t in previous.terms]
    cols = [m.variable_mapping[t.scalar_term.variable_index] for t in previous.terms]
    coefs = [0.0 for t in previous.terms]
    change_matrix_coefficients!(m, rows, cols, coefs)

    # Next, set the new constraint function terms
    rows = [constraint_indices[t.output_index] for t in replacement.terms]
    cols = [m.variable_mapping[t.scalar_term.variable_index] for t in replacement.terms]
    coefs = [t.scalar_term.coefficient for t in replacement.terms]
    change_matrix_coefficients!(m, rows, cols, coefs)
end

MOI.canset(m::LinQuadOptimizer, ::MOI.ConstraintFunction, ::Type{VLCI{S}}) where {S <: VecLinSets} = true
function MOI.set!(m::LinQuadOptimizer, attr::MOI.ConstraintFunction, c::VLCI{S}, replacement::VecLin) where {S <: VecLinSets}
    constraint_indices = m[c]
    previous = MOI.get(m, attr, c)
    MOI.Utilities.canonicalize!(previous)
    replacement = MOI.Utilities.canonical(replacement)

    if matching_sparsity_pattern(previous, replacement)
        replace_with_matching_sparsity!(m, previous, replacement, constraint_indices)
    else
        replace_with_different_sparsity!(m, previous, replacement, constraint_indices)
    end

    for i in eachindex(constraint_indices)
        row = constraint_indices[i]
        change_rhs_coefficient!(m, row, -replacement.constants[i])
        m.constraint_constant[row] = replacement.constants[i]
    end
end


MOI.candelete(m::LinQuadOptimizer, c::VLCI{<:VecLinSets}) = MOI.isvalid(m, c)
function MOI.delete!(m::LinQuadOptimizer, c::VLCI{<:VecLinSets})
    deleteconstraintname!(m, c)
    dict = constrdict(m, c)
    # we delete rows from largest to smallest here so that we don't have
    # to worry about updating references in a greater numbered row, only to
    # modify it later.
    for row in sort(dict[c], rev=true)
        delete_linear_constraints!(m, row, row)
        deleteat!(m.constraint_primal_solution, row)
        deleteat!(m.constraint_dual_solution, row)
        deleteat!(m.constraint_constant, row)
        # shift all the other references
        shift_references_after_delete_affine!(m, row)
    end
    delete!(dict, c)
end


#=
    Constraint set of Linear function
=#

MOI.canget(m::LinQuadOptimizer, ::MOI.ConstraintSet, ::Type{VLCI{S}}) where S <: VecLinSets = true
function MOI.get(m::LinQuadOptimizer, ::MOI.ConstraintSet, c::VLCI{S}) where S <: VecLinSets
    constraint_indices = m[c]
    S(length(constraint_indices))
end

#=
    Constraint function of Linear function
=#

MOI.canget(m::LinQuadOptimizer, ::MOI.ConstraintFunction, ::Type{VLCI{S}}) where S <: VecLinSets = true
function MOI.get(m::LinQuadOptimizer, ::MOI.ConstraintFunction, c::VLCI{<: VecLinSets})
    ctrs = m[c]
    n = length(ctrs)
    constants = Float64[]
    terms = MOI.VectorAffineTerm{Float64}[]
    for i in 1:n
        rhs = get_rhs(m, ctrs[i])
        push!(constants, -rhs)
        # TODO more efficiently
        colidx, coefs = get_linear_constraint(m, ctrs[i])
        for (column, coefficient) in zip(colidx, coefs)
            push!(terms, MOI.VectorAffineTerm{Float64}(
                    i,
                    MOI.ScalarAffineTerm{Float64}(
                        coefficient, m.variable_references[column]
                    )
                )
            )
        end
    end
    return MOI.VectorAffineFunction(terms, constants)
end
