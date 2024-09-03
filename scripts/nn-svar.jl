using Lux, Optimisers, Zygote
using CairoMakie
using Printf, Random, Statistics, LinearAlgebra

PLOTSDIR = mkpath(joinpath(dirname(Base.active_project()), "plots"))

rng = Xoshiro()
Random.seed!(rng, 31415)

# ENV["JULIA_DEBUG"] = Main
# ENV["JULIA_DEBUG"] = false

## Generate some artificial data
T = 100 
k = 2       # Number of yₜ variables 

# DGP is a VAR(1) process
z0 = 0.1 * randn(rng, Float32, k)
A1 = [0.7 0.3; 0.1 0.8]

z = zeros(Float32, k, T)
z[:, 1] = z0
e = 0.1 * randn(rng, Float32, k, T)

γ = zeros(Float32, 2)

# Some common factor that will be ignored in the estimated VAR
f = zeros(Float32, k, T) 
ef = 0.05 * randn(rng, Float32, k, T)

for t in 2:T 
    # f[:, t] = 0.5f[:, t-1] + ef[:, t]
    z[:, t] = γ + A1 * z[:, t-1] + f[:, t] + e[:, t]
end

x = z[:, begin:end-1]
y = z[:, begin+1:end]

# Check generated data
fig = Figure(size=(950, 650))
ax1 = Axis(fig[1, 1]; ylabel=L"y_{1t}")
l1 = lines!(ax1, y[1, :])
axislegend(ax1, [l1], ["Actual Data"])
ax2 = Axis(fig[2, 1]; ylabel=L"y_{2t}")
l1 = lines!(ax2, y[2, :])
fig


## VAR model 

Z = [ones(1, T-1); x]
Y = y
β_ols = kron((Z * Z') \ Z, I(k)) * vec(Y)

γ_ols  = β_ols[1:2]
A1_ols = reshape(β_ols[3:end], 2, 2)


## --- Functions for the script --- 

# We define a custom loss function according to the DTVP-SVAR slides objective function. 
# Lux.jl: 
# The function must take 4 inputs – model, parameters, states and data. The function must
# return 3 values – loss, updated_state, and any computed statistics (usually an empty NamedTuple)
# λ is the regularization penalty, for now has fixed value. Need to implement CV prediction to calibrate...
function lossnn(model, ps, st, data; λ=100)
    # Get the data
    x, y = first(data), last(data)

    # Get the trajectory of βₜ
    β, st_ = model(x, ps, st)

    # Disentangle SVAR coefficients
    γ = @view β[1:2, :]
    ϕ = @view β[3:6, :]
    α = @view β[7:9, :]

    # @debug "Sizes:" size(β) size(y) typeof(β) typeof(y)

    k, T = size(y)
    loss = zero(eltype(y))
    pen  = zero(eltype(y))
    Φ′   = zero(eltype(y))
    λ    = 100              
    for t in 1:T 
        # VAR's first lag matrix
        Φ = reshape(ϕ[:, t], k, k)
        # Predicted values from VAR model
        yhat = γ[:, T] + Φ * x[:, t]

        # Compute the structural contemporaneous effects matrix (B₀)
        A = [α[1, t] 0; α[2, t] α[3, t]]
        # This is the precision matrix Λ = Σᵤₜ⁻¹
        Λ = A' * A  

        # Penalization to smoothness in the AR1 coefficients
        if t > 1
            pen = sum(abs2, Φ - Φ′)
            # @debug pen
        end
        Φ′ = Φ

        # Compute loss function
        loss += (-log(det(Λ)) + (yhat - y[:, t])' * Λ * (yhat - y[:, t]) + λ * pen) / 2T
    end

    # loss, st_, (; y_pred=yhat)
    loss, st_, (; )
end

# Define a training function for the TrainState object that contains the MLP and optimizer
function train!(tstate::Training.TrainState, data, epochs; λ=100)

    # Vector for storing training loss
    lossfn = zeros(epochs)

    # Closure with the loss function and penalization hyper-parameter
    mlploss = (model, ps, st, data) -> lossnn(model, ps, st, data; λ)

    # Train for a number of epochs
    for epoch in 1:epochs
        # Automatically train a step with all the data
        _, loss, _, tstate = Training.single_train_step!(AutoZygote(), mlploss, data, tstate)
        
        # Save the training loss and print status
        lossfn[epoch] = loss
        if epoch % 5 == 1 || epoch == epochs
            @printf "Epoch: %3d \t Loss: %.5g \n" epoch loss
        end
    end

    return tstate, lossfn
end

# Helper to compute ŷ values from the Chain model evaluated at x
# Needs the Training.TrainState object to evaluate the model
function getpredvals(model, tstate, x)
    # Compute TVC
    β, st_ = model(x, tstate.parameters, tstate.states)
    γ = β[1:2, :]
    ϕ = β[3:6, :]
    α = β[7:9, :]   # Cholesky decomposition parameters (not used here)

    # Compute ŷ
    T = size(x, 2)
    yhat = zeros(eltype(x), k, T)
    for t in 1:T 
        Φ = reshape(ϕ[:, t], 2, 2)
        yhat[:, t] = γ[:, T] + Φ * x[:, t]
    end

    yhat
end


## Defining the βₜ neural network 

# State vector for the MLP is just Xₜ (same as for the VAR). It could include more
# factors. Input size is the dimension of Xₜ
model = Chain(
    Dense(k => 10, relu), 
    Dense(10 => 9)
)

# Adam optimizer
opt = Adam(0.03f0) 

# This is required to evaluate and train the model in Lux. Initializes the weights
tstate = Training.TrainState(rng, model, opt)

# Compute the loss with current parameters (initial loss value)
l_, _, nt_ = lossnn(model, tstate.parameters, tstate.states, (x, y))

## Train the model

# Train the model!
tstate, lossfn = train!(tstate, (x, y), 200)

## Inspect the model 

# Get predicted values
yhat = getpredvals(model, tstate, x)

## Plot actual vs predicted

fig = Figure(size=(950, 650))
ax1 = Axis(fig[1, 1]; ylabel=L"y_{1t}")
l1 = lines!(ax1, y[1, :])
l2 = lines!(ax1, yhat[1, :])
axislegend(ax1, [l1, l2], ["Actual Data", "Predictions"])
ax2 = Axis(fig[2, 1]; ylabel=L"y_{2t}")
l1 = lines!(ax2, y[2, :])
l2 = lines!(ax2, yhat[2, :])

save(joinpath(PLOTSDIR, "actual_predicted.png"), fig)
fig


## Loss function 

fig = Figure(size=(950, 650))
ax = Axis(fig[1, 1]; ylabel="Loss function", xlabel="Number of epochs")
lines!(ax, lossfn)

save(joinpath(PLOTSDIR, "training_loss.png"), fig)
fig

## Time-varying coefficients

β, st_ = model(x, tstate.parameters, tstate.states)

fig = Figure(size=(950, 650))

ax1 = Axis(fig[1, 1]; ylabel=L"\phi_{11}")
l1 = lines!(ax1, β[3, :])
l2 = hlines!(ax1, A1[1, 1], linestyle=:dash, color=:red)
l3 = hlines!(ax1, A1_ols[1, 1], linestyle=:dash, color=:green)

ax2 = Axis(fig[1, 2]; ylabel=L"\phi_{12}")
lines!(ax2, β[5, :])
hlines!(ax2, A1[1, 2], linestyle=:dash, color=:red)
hlines!(ax2, A1_ols[1, 2], linestyle=:dash, color=:green)

ax3 = Axis(fig[2, 1]; ylabel=L"\phi_{21}")
lines!(ax3, β[4, :])
hlines!(ax3, A1[2, 1], linestyle=:dash, color=:red)
hlines!(ax3, A1_ols[2, 1], linestyle=:dash, color=:green)

ax4 = Axis(fig[2, 2]; ylabel=L"\phi_{22}")
lines!(ax4, β[6, :])
hlines!(ax4, A1[2, 2], linestyle=:dash, color=:red)
hlines!(ax4, A1_ols[2, 2], linestyle=:dash, color=:green)

ax5 = Axis(fig[1, 3]; ylabel=L"\gamma_{1}")
lines!(ax5, β[1, :])
hlines!(ax5, γ[1], linestyle=:dash, color=:red)
hlines!(ax5, γ_ols[1], linestyle=:dash, color=:green)

ax6 = Axis(fig[2, 3]; ylabel=L"\gamma_{2}")
lines!(ax6, β[2, :])
hlines!(ax6, γ[2], linestyle=:dash, color=:red)
hlines!(ax6, γ_ols[2], linestyle=:dash, color=:green)

Legend(fig[3,2], [l1, l2, l3], ["NN-TVC", "DGP (true)", "VAR(1)"], orientation=:horizontal, position=:rb, tellwidth=false)

save(joinpath(PLOTSDIR, "tvc_params.png"), fig)
fig


