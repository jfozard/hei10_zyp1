
using ArgParse
using DifferentialEquations
using DiffEqCallbacks
using LinearAlgebra
using Random
using SparseArrays
using StatsBase
#using NLopt
#using StatsPlots
using FiniteDiff
#using Plots
using Distributions
#using GLM


# Piecewise linear functions

# Piecewise linear interpolation at multiple positions (specialized for Float64)
function interp1d(x::Array{Float64,1}, x0::Array{Float64,1}, y0::Array{Float64,1})
    # x - position to evaluate function
    # x0 - positions of discontinuities in function
    # y0 - values of function on each region: y = y0[i] for x0[i] < x < x0[i+1]


    y = zeros(Float64, size(x))
    jj = 1
    for ii in 1:length(y)
        if x0[1]<=x[ii]<x0[end]
            jj = findnext(val->val>x[ii],x0,jj)-1
            y[ii] = y0[jj] + (y0[jj+1]-y0[jj])/(x0[jj+1]-x0[jj])*(x[ii]-x0[jj])
        end
    end
    return y
end

# Piecewise linear interpolation at a single position
function interp1d(x::Number, x0::Array, y0::Array)
    # x - position to evaluate function
    # x0 - positions of discontinuities in function slope
    # y0 - values of function at each discontinuity. Function takes the value 0 outside the range [x0[1], x0[length(x0)]]

    jj = 1
    y = 0.0
    if x0[1]<=x<x0[end]
        jj = findnext(val->val>x,x0,jj)-1
        y = y0[jj] + (y0[jj+1]-y0[jj])/(x0[jj+1]-x0[jj])*(x-x0[jj])
    end
    return y
end


# Piecewise constant interpolation
function piecewise(x::Number, x0::Array, y0::Array)
    # x - position to evaluate function
    # x0 - positions of discontinuities in function
    # y0 - values of function on each region: y = y0[i] for x0[i] < x < x0[i+1]

    jj = 1
    y = 0.0
    if x0[1]<=x<x0[end]
        jj = findnext(val->val>x,x0,jj)-1
        y = y0[jj]
    end
    return y
end


# Inverse of a piecewise constant density function
function piecewise_density_inv(x0::Array{T,1}, y0::Array{T,1}) where {T <: Number}
    csum = zeros(T, size(x0))
    for i in 2:length(x0)
        csum[i] = (x0[i] - x0[i-1])*y0[i-1] + csum[i-1]
    end
    csum = csum / csum[end]
    csum, x0
end


# Inverse of a piecewise linear density function
function linear_density_inverse(x0::Array{T,1}, y0::Array{T,1}) where {T <: Number}
    csum = zeros(T, size(x0))
    for	i in 2:length(x0)
    	csum[i] = (x0[i] - x0[i-1])*(y0[i-1] + y0[i])/2	+ csum[i-1]
    end
    csum_t = csum[end]
    function inverse(x::T)
        z = x*csum_t
        jj = findnext(val->val>z, csum, 1)-1
        z = z - csum[jj]
        y = y0[jj]
        dy = (y0[jj+1] - y0[jj])/(x0[jj+1]-x0[jj])
        if abs(dy)>1e-12
            return x0[jj]+((sqrt(y*y+2*z*dy)-y)/dy)
        elseif abs(y)>1e-12
            return x0[jj]+z/y
        else
            return x0[jj]
        end
    end
    inverse
end

# RI HEI10 escape rate function (\beta(C)*C in equation (2) )
function betaC(C, K, n_c)
    C ./ (1 .+ (C / K) .^ n_c)
end

# RI HEI10 escape rate function \beta(C)*C (evaluated in place)
function betaC!(dC, C, K, n_c)
    @. dC =  C ./ (1 .+ (C ./ K) .^ n_c)
end

# Derivative of (\beta(C)*C) wrt C
function d_betaC(C::Float64, K::Float64, n_c::Float64)::Float64
    1 ./ (1.0 .+ (C ./ K) .^ n_c) .- C ./ (1.0 .+ (C ./ K) .^ n_c) .^ 2 .* (n_c ./ K) .* (C ./ K) .^ (n_c - 1)
end

# Derivative of (\beta(C)*C) wrt C (evaluated in place)
function d_betaC!(dbC, C, K, n_c)
    @. dbC = 1 ./ (1.0 .+ (C ./ K) .^ n_c) .- C ./ (1.0 .+ (C ./ K) .^ n_c) .^ 2 .* (n_c ./ K) .* (C ./ K) .^ (n_c - 1)
end

### Distributions of chromosome lengths

Lm = Vector{Float64}([37.5, 40.4, 45.7, 53.8, 59.8])
Ls = Vector{Float64}([4.93, 5.06, 5.86, 7.18, 7.13])

function H(t)
    0.5 .* (sign.(t) .+ 1)
end


function Hs(t; s=10)
    0.5 .* (tanh.(t.*s) .+ 1)
end

function dHs(t; s=10)
    0.5 *s .* ( 1.0./ cosh.(t.*s).^2)
end

function basic_ode_smooth!(dyy::Array{Float64,1}, yy::Array{Float64,1}, p, t::Float64)
    m_array,           # Number of SC spatial compartments for discretization (array)
    N_array,           # Number of RIs (array)
    Q,           # Number of SCs
    K,           # Hill threshold
    K_u,         # Threshold for nucleoplasmic recycling
    nodes_array,       # Indices of SC spatial compartments containing RIs
    a_SC,       # Rate of HEI10 absorption onto SC from pool
    b_SC,        # Rate of HEI10 escape from SC to pool
    a_nodes, # Rate of HEI10 absorption from pool to nodes
    D,           # RI Diffusion coefficient on SC (D_c)
    dL_array,          # Length of each spatial compartment (array)
    betaC!,      # beta(C)*C function (passed to ODE for efficiency)
    d_betaC,     # derivative of beta(C)*C function wrt C
    gamma,         # Hill coefficient (\gamma)
    alpha,     # RI HEI10 absorption rate from SC (\alpha)
    beta,     # RI HEI10 escape rate back onto SC (\beta)
    beta_2,    # RI HEI10 escape rate into pool (C-dependent)
    beta_3,    # RI HEI10 escape rate into pool (const)
    pool_decay,
    bC,          # Placeholder to re-use vector
    A = p

    @inbounds pool = yy[end]

    @inbounds dyy[end] = 0

    offset = 0
    for q = 1:Q

        @inbounds m = m_array[q]
        @inbounds N = N_array[q]
        @inbounds nodes = nodes_array[q]
        @inbounds dL = dL_array[q]

        # Extract variables from ODE State vector

    # Extract variables from ODE State vector
        @inbounds y = @view yy[offset+1:offset+m] # HEI10 concentration on the SC (c_i, 1<=i<=m)
        @inbounds dy = @view dyy[offset+1:offset+m]
        @inbounds C = @view yy[offset+m+1:offset+m+N] # HEI10 amounts at the RH (C_j 1<=j<=N)
        @inbounds dC = @view dyy[offset+m+1:offset+m+N]


        h = D/dL/dL

        @inbounds dy[1] = a_SC*pool - b_SC * y[1] + h * (y[2] - y[1])
        @inbounds @simd for i = 2:(m-1)
            dy[i] = a_SC*pool - b_SC * y[i] +
                    h * (y[i+1] - 2 * y[i] + y[i-1])
        end

        @inbounds dy[m] = a_SC*pool - b_SC * y[m] + h * (y[m-1] - y[m])

        # Calculate escape of HEI10 back onto SC.

        bbC = @view bC[1:N]
        betaC!(bbC, max.(C,0.0), K, gamma)

        # Calculate net fluxes from SC to RIs, and update ODE RHS for appropriate SC compartment
        @inbounds for j = 1:N
            A[j] = (alpha * y[nodes[j]]  - beta * bC[j])*Hs(C[j] - K_u)
            dyy[nodes[j]+offset] = dyy[nodes[j]+offset] - A[j] / dL
        end

        # Rate of change of HEI10 amounts at the RIs
        @. dC = A[1:N] .- beta_2 .* bC[1:N] .*Hs(C .- K_u) .- beta_3.*Hs(C .- K_u) .* C + a_nodes*pool.*Hs(C.-K_u)

        @inbounds dyy[end] = dyy[end]- a_SC*pool*m*dL - a_nodes*pool*sum(Hs(C.-K_u)) + b_SC*sum(y)*dL + beta_2 *sum(bC[1:N].*Hs(C .- K_u) ) + beta_3 *sum(C .* Hs(C .- K_u))
        offset += (m + N)
    end
    @inbounds dyy[end] = dyy[end] - pool_decay*pool
end


function jac_smooth!(J, yy, p, t)
    m_array,           # Number of SC spatial compartments for discretization (array)
    N_array,           # Number of RIs (array)
    Q,           # Number of SCs
    K,           # Hill threshold
    K_u,         # Threshold for uptake
    nodes_array,       # Indices of SC spatial compartments containing RIs
    a_SC,       # Rate of HEI10 absorption onto SC from pool
    b_SC,        # Rate of HEI10 escape from SC to pool
    a_nodes, # Rate of HEI10 absorption from pool to nodes
    D,           # RI Diffusion coefficient on SC (D_c)
    dL_array,          # Length of each spatial compartment (array)
    betaC!,      # beta(C)*C function (passed to ODE for efficiency)
    d_betaC,     # derivative of beta(C)*C function wrt C
    gamma,         # Hill coefficient (\gamma)
    alpha,     # RI HEI10 absorption rate from SC (\alpha)
    beta,     # RI HEI10 escape rate back onto SC (\beta)
    beta_2,    # RI HEI10 escape rate into pool (C-dependent)
    beta_3,    # RI HEI10 escape rate into pool (const)
    pool_decay,
    bC,          # Placeholder to re-use vector
    A = p

    J[end, end] = -pool_decay
    offset = 0

    pool = yy[end]
    for q in 1:Q
        nodes = nodes_array[q]
        m = m_array[q]
        N = N_array[q]
        dL = dL_array[q]
        r = D / dL / dL
        C = @view yy[offset+m+1:offset+m+N]
        y = @view yy[offset+1:offset+m]
        J[offset+1, offset+1] = - b_SC - r
        J[offset+1, offset+2] = r
        @inbounds for j = 2:m-1
            J[offset+j, offset+j-1] = r
            J[offset+j, offset+j] = - b_SC - 2 * r
            J[offset+j, offset+j+1] = r
        end
        @inbounds J[offset+m, offset+m-1] = r
        @inbounds J[offset+m, offset+m] = - b_SC - r
        @inbounds for j=1:m
            J[offset+j, end] = a_SC
            J[end, offset+j] = b_SC*dL
        end
        dbC::Float64 = 0.0
        idx::Int64 = 0
        bbC = @view bC[1:N]
        betaC!(bbC, max.(C,0.0), K, gamma)
        @inbounds for j = 1:N
            dbC = d_betaC(max(0.0, C[j]), K, gamma)
	    Hs_j = Hs(C[j]-K_u)
   	    dHs_j = dHs(C[j]-K_u)	

            J[offset+m+j, offset+m+j] = a_nodes*pool*dHs_j +alpha*y[nodes[j]]*dHs_j -(beta + beta_2)*Hs_j * dbC - (beta + beta_2)*dHs_j*bbC[j] - beta_3*Hs_j -beta_3*dHs_j*C[j]
            J[offset+nodes[j], offset+m+j] = beta * Hs_j *dbC / dL + beta * dHs_j * bbC[j] /dL - alpha*dHs_j/dL*y[nodes[j]]
            J[offset+m+j, offset+nodes[j]] = alpha*Hs_j
            J[offset+nodes[j], offset+nodes[j]] -= alpha*Hs_j / dL
            J[offset+m+j, end] = a_nodes*Hs_j
            J[end, offset+m+j] = beta_2*dbC*Hs_j + beta_2*bbC[j]*dHs_j + beta_3*Hs_j + beta_3*C[j]*dHs_j -a_nodes*pool*dHs_j
        end
        J[end, end] = J[end, end] - m*a_SC*dL - sum(Hs(C.-K_u))*a_nodes
        offset += m+N
    end
end


