###################################
###     Load LHI Parameters     ###
###        by Cooper Niu        ###
###        March 26, 2024       ###
###################################

const g = big(0.64) # Gauge Coupling Constants
const mpl = big(2.4e18) # Reduced Planck Mass [GeV]
const coeff = BigFloat[0.05, 1.0, 1.0] # Fiducial Choice of the Potential Parameters

function calc_model(f_pt, beta_pt)
    f = f_pt
    beta = beta_pt
    mu = 10^(0.499 * log10(f_pt) + 1.065)
    return (mu, beta, f)
end	

test_model = calc_model(10^18.2, 10^12)

using QuadGK

function V_higgs(h, model, coeff, reparam)
    c0, c2, c4 = coeff
    mu, beta, f = model
    mu /= mpl
    f /= mpl
    higgs_angle = h / f
    
    if !reparam
        V = mu^4 * (c0 + c2 * sin(higgs_angle)^2 + c4 * sin(higgs_angle)^4)
    else
        eps = h
        V = mu^4 * c0
    end
    return V
end

function Vh(h, model, coeff, reparam)
    c0, c2, c4 = coeff
    mu, beta, f = model
    mu /= mpl
    f /= mpl
    higgs_angle = h / f
    
    if !reparam
        Vh = mu^4 * (2 * c2 * sin(higgs_angle) * cos(higgs_angle) + 4 * c4 * cos(higgs_angle) * sin(higgs_angle)^3) / f
    else
        eps = h
        Vh = -2 * c2 * (mu^4 / f) * eps * π
    end
    return Vh
end

function Vhh(h, model, coeff, reparam)
    c0, c2, c4 = coeff
    mu, beta, f = model
    mu /= mpl
    f /= mpl
    higgs_angle = h / f
    
    if !reparam
        Vhh = (2 * mu^4 / f^2) * ((c2 + c4) * cos(2 * higgs_angle) - c4 * cos(4 * higgs_angle))
    else
        eps = h
        Vhh = 2 * c2 * mu^4 / f^2
    end
    return Vhh
end

function hubble(h, model, coeff, reparam)
    hub2 = V_higgs(h, model, coeff, reparam) / 3
    hub = sqrt(hub2)
    
    return hub
end

function psi_bg(h, model, coeff, reparam)
    c0, c2, c4 = coeff
    mu, beta, f = model
    mu /= mpl
    f /= mpl
    lamb = g * beta
    
    hub = hubble(h, model, coeff, reparam)
    vh = Vh(h, model, coeff, reparam)
    
    if !reparam
        psi_bg = (-f^2 * vh / (3 * g * h * hub * lamb^2))^(1/3)
    else
        eps = h
        psi_bg = ((2 * c2 * mu^4) / (3 * g * lamb^2 * hub))^(1/3) * eps^(1/3)
    end

    return psi_bg
end

function m_eff(h, model, coeff, reparam)
    hub = hubble(h, model, coeff, reparam)
    psi = psi_bg(h, model, coeff, reparam)
    m = g * psi / hub

    return m
end

function N_integrand(h, model, coeff, reparam)
    mu, beta, f = model
    mu /= mpl
    f /= mpl
    lamb = g * beta
    
    hub = hubble(h, model, coeff, reparam)
    psi = psi_bg(h, model, coeff, reparam)
    m_psi = m_eff(h, model, coeff, reparam)
    
    if !reparam
        dh = (f^2 / lamb^2) * (hub / h) * (2 * m_psi + 2 / m_psi)
        dN = hub / dh
    else
        dN = (π * lamb)^2 * (h - 1) * (2 * m_psi + 2 / m_psi)^(-1)
    end
    
    return dN
end

function delta_N(h_i, model, coeff, reparam)
    mu, beta, f = model
    mu /= mpl
    f /= mpl
    lamb = g * beta
    
    if !reparam
        h_f = π * f

        res, _ = quadgk(h -> N_integrand(h, model, coeff, reparam), h_i, h_f)
    else
        eps_i = h_i
        eps_f = 0 
        
        res, _ = quadgk(eps -> N_integrand(eps, model, coeff, reparam), eps_i, eps_f)
    end
    
    return res
end


function solve_hc_higgs(model, coeff, N_ref, reparam)
    """
    Solve for the horizon-crossing value of the higgs field
    -------------------------------------------------------
    model: benchmark models
    coeff: coefficients for the full little higgs potential
    V_type: potential types (minimal/full)
    N_ref: number of e-folds before the end of inflation (50 or 60)
    """
    mu, beta, f = model
    mu /= mpl
    f /= mpl
    
    maxiter = 26
    n_iter = 1
    tol = 0.1
    
    if !reparam
        x0 = 0.9
        error = delta_N(x0 * π * f, model, coeff, reparam) - N_ref
        while abs(error) > tol && n_iter <= maxiter
            while error > 0
                x0 += 0.9 * 0.1^n_iter
                n_iter += 1
                error = delta_N(x0 * π * f, model, coeff, reparam) - N_ref
            end
            
            while error < 0
                x0 -= 0.1^n_iter
                error = delta_N(x0 * π * f, model, coeff, reparam) - N_ref
            end
        end
        hc_higgs_angle = x0 * π * f
    else
        y0 = 0.1
        print
        error = delta_N(y0, model, coeff, reparam) - N_ref
        while abs(error) > tol && n_iter <= maxiter
            while error > 0
                y0 = y0 - 0.1^n_iter + 0.1^(n_iter+1)
                n_iter += 1
                error = delta_N(y0, model, coeff, reparam) - N_ref
            end
            
            while error < 0
                y0 += 0.1^n_iter
                error = delta_N(y0, model, coeff, reparam) - N_ref
            end
        end
        hc_higgs_angle = y0
    end
    
    return hc_higgs_angle
end


function load_param(model, coeff, N_ref, print_param = false)
    """
    Load The Coefficients For A Given Set of Benchmark Values
    ---------------------------------------------------------
    model: benchmark models
    coeff: full potential coefficients choice
    V_type: potential types (minimal/full)
    N_ref: reference number of e-folds before the end of inflation (50 or 60)
    ---------------------------------------------------------
    Return:
    hbar: higgs field value when horizon crossing
    vhh: V''(h) f
    Hubble: hubble parameter when horizon crossing
    psi: gauge field value when horizon crossing
    m: dimensionless effective gauge mass
    Lambda：a constant parameter (lambda^2*psi^2/f^2)
    """

    mu, beta, f = model
    mu /= mpl
    f /= mpl
    N = delta_N(0.500001 * π * f, model, coeff, false)
    
    if N > 1e6
        #println("We expand the potential")
        reparam = true
        epsilon = solve_hc_higgs(model, coeff, N_ref, reparam)
        hbar = (1 - epsilon) * π * f
        
        vhh = Vhh(epsilon, model, coeff, reparam)
        Hubble = hubble(epsilon, model, coeff, reparam)
        psi = psi_bg(epsilon, model, coeff, reparam)
        m = m_eff(epsilon, model, coeff, reparam)
        lamb = g * beta
        Lambda = (lamb^2 / f^2) * psi
        Gamma_c = Lambda * hbar * m
        Psi_c = 3 * Lambda * m * psi
        Beta_c = sqrt(2) * Lambda * hbar * m
        nu = sqrt(2) * Lambda * hbar
        
    else
        reparam = false
        
        hbar = solve_hc_higgs(model, coeff, N_ref, reparam)
        vhh = Vhh(hbar, model, coeff, reparam)
        Hubble = hubble(hbar, model, coeff, reparam)
        psi = psi_bg(hbar, model, coeff, reparam)
        m = m_eff(hbar, model, coeff, reparam)
        
        lamb = g * beta
        Lambda = (lamb^2 / f^2) * psi
        Gamma_c = Lambda * hbar * m
        Psi_c = 3 * Lambda * m * psi
        Beta_c = sqrt(2) * Lambda * hbar * m
        nu = sqrt(2) * Lambda * hbar
    end
    
    # Four Regions
    #region1 = Lambda * hbar * m_psi
    #region2 = m_psi
    #region3 = sqrt(2) * m_psi / (Lambda * Hubble)
    
    
    # Initial Condition for the scalar modes Equations of Motion
    A = 1.0
    B = nu * A / sqrt(Beta_c^2 + nu^2)
    C = Beta_c * A / sqrt(Beta_c^2 + nu^2)
    x0 = sqrt(2) * m * Lambda * hbar * 100 # We add two more orders of magnitudes to ensure the WKB methods work well for the initial conditions
    
    A_prime = 1
    B_prime = 1
    C_prime = 1

    initial0 = [A, A_prime, B, B_prime, C, C_prime]
    
    # Instability
    tensor_low = 2 * m + 1 / m - sqrt(2 * m^2 + 2 + 1 / m^2)
    tensor_up = 2 * m + 1 / m + sqrt(2 * m^2 + 2 + 1 / m^2)

    # Print Input Parameters
    if print_param
        println("##==========================================##")
        println("##          Solve Little Higgs ODE          ##")
        println("##==========================================##")
        println(" ")
        println("#================ Model Params ==============#")
        println(" ")
        println("Strong Scale(μ):           ", mu, "[GeV]")
        println("CS coupling(β):            ", beta)
        println("Decay Constant(f):         ", f, "[GeV]")
        println("")
        println("#============ Scalar Modes Params ===========#")
        println(" ")
        println("N of e-folds:              ", N)
        println("Higgs θ:                   ", hbar / f)
        println("Higgs Field:               ", hbar)
        println("H:                         ", Hubble)
        println("V,hh:                      ", vhh)
        println("ψ:                         ", psi)
        println("m_ψ:                       ", m)
        println("Λ:                         ", Lambda)
        println(" ")
        println("#=============== Coefficients ===============#")
        println(" ")
        println("Psi_c:                     ", Psi_c)
        println("Beta_c:                    ", Beta_c)
        println("Gamma_c:                   ", Gamma_c)
        println("nu:                        ", nu)
        println(" ")
        println("#============ Initial Conditions ============#")
        println(" ")
        println("h:                         ", A)
        println("φ:                         ", B)
        println("z:                         ", C)
        println("x0:                        ", x0)
        println(" ")
        println("#=========== Range of Instability ===========#")
        println(" ")
        println("Scalar Modes (m_ψ^2 < 0)   ", sqrt(m))
        println("Tensor Modes:              [", tensor_low, ",", tensor_up, "]")
        println(" ")
        println("##==========================================##")
        println(" ")
        #println("m_psi : ", m)
        #println("Sqrt(2)*m_psi/H*Lambda: ", sqrt(2) * m / (Hubble * Lambda))
    end
    
    return N, hbar, vhh, Hubble, psi, m, Lambda, x0, initial0
end
