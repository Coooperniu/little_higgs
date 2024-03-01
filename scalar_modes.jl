#----------------------------#
# Constants and Coefficients #
#----------------------------#

const g = BigFloat(0.64)  # Gauge Coupling Constants
const mpl = BigFloat(2.4e18)  # Reduced Planck Mass [GeV]
const coeff = [0.05, 1, 1] .|> BigFloat  # Fiducial Choice of the Potential Parameters

# The ordering of the values are ['mu' (GeV), 'beta', 'f' (GeV)]
Model_A = [7.8e6, 3.2e6, 5.3e11] .|> BigFloat
Model_B = [1.1e8, 8.6e5, 1e14] .|> BigFloat
Model_C = [3.5e9, 1.5e5, 1e17] .|> BigFloat
Model_D = [1.1e8, 2e6, 1e14] .|> BigFloat
Model_E = [3.7e9, 5e7, 1e17] .|> BigFloat
Model_F = [3.5e9, 1e6, 1e17] .|> BigFloat
Model_G = [7.8e6, 3.2e6, 5.3e11] .|> BigFloat

# Benchmark Parameters for Model_E
mu0 = Model_E[1]/mpl
beta0 = Model_E[2]
f0 = Model_E[3]/mpl　

#----------------------------#
#   ODE Solver Parameters    #
#----------------------------#

if length(ARGS) < 3
    println("Usage: julia file.jl t1 t2 abstol reltol")
    exit(1)
end

tspan_input = (parse(BigFloat, ARGS[1]), parse(BigFloat, ARGS[2]))
abstol_input = parse(BigFloat, ARGS[3])
reltol_input = parse(BigFloat, ARGS[4])

println("------------------------------------------------")
println("                 Scalar Modes ODE               ")
println("------------------------------------------------")
println("")
println("Time Span: $(ARGS[1], ARGS[2])")
println("abstol:    $ARGS[3]")
println("reltol:    $ARGS[4]")
println("")
println("------------------------------------------------")
println("")

#----------------------------#
# Constants and Coefficients #
#----------------------------#

param = [3.121080192438486081550179297471783836263718303027941183241874459052646640630746e+08,
    0.130899693861613803897500260632283739274896856733686163949974802434577118136583,
    6.507503472222222243828015321136643745914603198812290605855395359345549624539073e-33,
    3.068353125554371473948489305808892279441523628142995247745208357018653168685175e-19,
    1.75776634383698095576950055182781419015454218777239023876538302211835499831307e-14,
    36663.65682249871355274111018201327689981707144214946272124133819908557228108921,
    10367.72775987303498420127663692935733963824580850858646107563029902490699903216] .|> BigFloat

u0 = BigFloat[1.0,
              0.0,
              2.727496617343380922529943412967894723911322296154469907423733565109550153413726e-05,
              0.0,
              0.9999999996280381100498429113021821165848074344082024189073783682287777821107523,
              0.0]

#----------------------------#
# Constants and Coefficients #
#----------------------------#
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

# Solve the differential equations

using Plots, Measures

#tspan = (1e+10, 1e-3)
alg = RadauIIA5()

@time begin
    prob = ODEProblem(
		scalar_modes!, 
		u0, 
		tspan_input, 
		param, 
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

timestamp = Dates.format(now(), "yy-mm-dd-HH-MM")
fig_name = "scalarODE_$timestamp.png"
plot_title = "Input Values: tspan=$(ARGS[1])-$(ARGS[2]), abstol=$(ARGS[3]), reltol=$(ARGS[4])"


plot(sol0.t, broadcast(abs,sol0[1,:]), label = "h", line=(2,:orange,:solid))
plot!(sol0.t, broadcast(abs,sol0[3,:]), label = "phi", line=(2,:blue,:solid))
plot!(sol0.t, broadcast(abs,sol0[5,:]), label = "z", line=(2,:green,:solid))

plot!(size=(1000,600),
    title = "plot_title"
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
println("Plot Saved!")

using DataFrames
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
println("Data Saved!")

