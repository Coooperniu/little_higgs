################################################
###       Scalar Modes ODEs Solver V2.0      ###
###                by Cooper Niu             ###
###               March 26, 2024             ###
################################################

function calc_model(f_pt, beta_pt)
    f = f_pt
    beta = beta_pt
    mu = 10^(0.499 * log10(f_pt) + 1.065)
    return (mu, beta, f)
end

#----------------------------#
#   ODE Solver Parameters    #
#----------------------------#

if length(ARGS) < 3
    println("Usage: f beta abstol reltol")
    println("=========================================")
    println("f:         Strong Coupling Constant [GeV]")
    println("beta:      CS Coupling Constant")
    println("abstol:    Absolute Tolerance for ODE solver")
    println("reltol:    Relative Tolerance for ODE solver")
    println("")
    exit(1)
end

f = parse(BigFloat, ARGS[1])
beta = parse(BigFloat, ARGS[2])
abstol_input = parse(BigFloat, ARGS[3])
reltol_input = parse(BigFloat, ARGS[4])

this_model = calc_model(f, beta)


include("load_param.jl")
params_tol = load_param(this_model, coeff, 60, true)

println("abstol:                    ", abstol_input)
println("reltol:                    ", abstol_input)
println("")
println("##==========================================##")
println(" ")

params = params_tol[1:7]
x0 = params_tol[8]
u0 = params_tol[9]


#-----------------------#
#       ODE Solver      #
#-----------------------#

using DifferentialEquations

function scalar_modes!(du, u, p, t)
    """
    The ODE system for the Scalar Modes
    ----------------------------------
    x: unscaled time steps
    var: differential parameters
    PARAM:
    simplified (boolean): whether to drop small coefficient terms or not
    """

    h, dh, phi, dphi, z, dz = u
    H, hbar, vhh, Hubble, psi, m, Lambda= p

    scale = 1.0

    dh2 = -(1 - 2/t^2 + vhh/(Hubble^2*t^2) + Lambda^2*hbar^2*m^2/(2*m^2+t^2)- (3*Lambda*m*psi)/t^2)*h*scale^2 -
          ((2*sqrt(2)*Lambda*m*hbar)/t^2)*z*scale^2 -
          ((sqrt(2)*Lambda*m*hbar)/t)*dz*scale +
          ((sqrt(2)*Lambda*m*hbar)/t)*(t^2/m^2+2)^(-1/2)*dphi*scale -
          (Lambda*hbar*sqrt(2*t^2/m^2 +4)*(m^3*t^2-m*(4*m^4+2*t^2*m^2 +t^4))/(2*m^2*t+t^3)^2)*phi*scale^2

    dphi2 = - (1-2/(2*m^2 + t^2)+2*m^2/t^2 + 6*m^2/(2*m^2+t^2)^2) * phi * scale^2 -
            (2*sqrt(2 + t^2/m^2)/t^2)*z * scale ^2 -
            ((sqrt(2)*Lambda*m*hbar)/t)*(t^2/m^2+2)^(-1/2) * dh * scale +
            (Lambda*hbar*m*sqrt(2*t^2/m^2+4)*(m^2*(2*m^2+t^2)+(4*m^4+2*t^2*m^2 +t^4))/(2*m^2*t+t^3)^2)*h*scale^2

    dz2 = - (3*sqrt(2)*Lambda*m*hbar/t^2)*h*scale^2 +
          (sqrt(2)*Lambda*m*hbar/t)*dh*scale -
          (1-(2-2*m^2)/t^2)*z*scale^2 -
          (2*sqrt(2+t^2/m^2)/t^2)*phi*scale^2

    du[1] = dh
    du[2] = dh2
    du[3] = dphi
    du[4] = dphi2
    du[5] = dz
    du[6] = dz2
end

using Plots, Measures

tspan = (x0, 1e-3)
alg = RadauIIA5()

println("ODE Solver Starts!")

@time begin
    prob = ODEProblem(
                scalar_modes!,
                u0,
                tspan,
                params,
                abstol = abstol_input,
                reltol = reltol_input
                )
    sol0 = solve(prob, alg, progress = true, maxiters = 1e8)
end

println("")
println("------------------------------------------------")
println("")

using Dates, CSV, DataFrames

println("Plot Maker Starts!")

timestamp = Dates.format(now(), "yy-mm-dd-HH-MM")
fig_name = "scalarODE_$timestamp.png"
plot_title = "Model($(ARGS[3]), $(ARGS[3])), abstol=$(ARGS[3]), reltol=$(ARGS[4])"


plot(sol0.t, broadcast(abs,sol0[1,:]), label = "h", line=(2,:orange,:solid))
plot!(sol0.t, broadcast(abs,sol0[3,:]), label = "phi", line=(2,:blue,:solid))
plot!(sol0.t, broadcast(abs,sol0[5,:]), label = "z", line=(2,:green,:solid))

plot!(size=(1000,600),
    title = plot_title,
    xlabel="x",
    ylabel="Amplitude",
    xscale=:log10,
    yscale=:log10,
    gridlinewidth = 2,
    minorgrid=true,
    xtickfont = font(20, "Times"),
    ytickfont = font(20, "Times"),
    xguidefont = font(20, "Times"),
    yguidefont = font(20, "Times"),
    legendfont = font(20, "Times"),
    margin = 10mm,
    dpi=300
    )

savefig(fig_name)
println("Plot Saved! ($fig_name)")

using DataFrames

println("Data Saver Starts!")


df = DataFrame(
        t = sol0.t,
        h = sol0[1,:],
        dh = sol0[2,:],
        phi = sol0[3,:],
        dphi = sol0[4,:],
        z = sol0[5,:],
        dz = sol0[6,:]
        #tspan = tspan_input,
        #abstol = [abstol_input],
        #reltol = [reltol_input]
)

csv_filename = "sol_$timestamp.csv"
CSV.write(csv_filename, df)
println("Data Saved! ($csv_filename)")

