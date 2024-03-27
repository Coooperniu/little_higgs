################################################
###       Scalar Modes ODEs Solver v1.0      ###
###               by Cooper Niu              ###
###              March 26, 2024              ###
################################################

#----------------------------#
# Constants and Coefficients #
#----------------------------#

const g = BigFloat(0.64)  # Gauge Coupling Constants
const mpl = BigFloat(2.4e18)  # Reduced Planck Mass [GeV]
const coeff = [0.05, 1, 1] .|> BigFloat  # Fiducial Choice of the Potential Parameters

#----------------------------#
#   ODE Solver Parameters    #
#----------------------------#

if length(ARGS) < 1
    println("Usage: julia file.jl abstol reltol")
    exit(1)
end

abstol_input = parse(BigFloat, ARGS[1])
reltol_input = parse(BigFloat, ARGS[2])

println("------------------------------------------------")
println("                 Scalar Modes ODE               ")
println("------------------------------------------------")
println("")
println("abstol:    $(ARGS[1])")
println("reltol:    $(ARGS[2])")
println("")
println("------------------------------------------------")
println("")

#----------------------------#
# Constants and Coefficients #
#----------------------------#

params = BigFloat[4.663589476727777010454030601414653685807325518180622338147111345686418857389183e+13,
		  1.308996938995747131665818111893410483996073404947916666666666666666666666666661,
		  5.353126572741465860953474352707015626366226837365858662107411480880443388516149e-33,
		  2.782927493683674973127632293141304564399294031363421858585864875016911292970139e-18,
		  8.935302817637800733882019448372579308152531670918528064983508799285487068345954e-18,
		  2.054884223993441884512485776938424452068217005932342524900705850726119178889823,
		  133.012270459095]

x0 = 50597.92342958567695625278340736627636704463552095318370490674388959507990903101

u0 = BigFloat[1.0, 1.0, 0.437581276709624719062164707868806212170521114642364399353896798623163412648814, 1.0, 0.8991788622255167556134386968333411803859867896292619301797523424330876690726461, 1.0]



#-------------------------#
#       ODE Solvers       #
#-------------------------#

using DifferentialEquations

function scalar_modes!(du, u, p, t)
    #=
    The ODE system for the Scalar Modes
    ----------------------------------
    x: unscaled time steps
    var: differential parameters
    PARAM:
    simplified (boolean): whether to drop small coefficient terms or not
    =#

    h, dh, phi, dphi, z, dz = u
    H, hbar, vhh, Hubble, psi, m, Lambda = p

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


# Solve the differential equations

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
    sol0 = solve(prob, alg, progress = true, maxiters = 1e9)
end

println("")
println("------------------------------------------------")
println("")


# Save the figure with a timestamp in the filename

using Dates, CSV, DataFrames

println("Plot Maker Starts!")

timestamp = Dates.format(now(), "yy-mm-dd-HH-MM")
fig_name = "scalarODE_$timestamp.png"
plot_title = "[Rodas5P]: abstol=$(ARGS[1]), reltol=$(ARGS[2])"

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

