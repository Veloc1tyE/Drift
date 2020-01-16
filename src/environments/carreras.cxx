// TODO: integrate physicsmodel solvers into environment
// TODO: Complete the simulation environment

#include <bout/physicsmodel.hxx>

#include <invert_laplace.hxx>
#include <invert_parderiv.hxx>
#include <inititialprofiles.hxx>
#include <bout/constants.hxx>

class ResistiveMHD : public PhysicsModel {    
    // 3D evolving fields
    Field3D psi, U, p_hat;
    
    // Derived 3D variables
    Field3D phi, J_z, p_0, Z, omega, V;
    
    // External coil fields
    Field3D U_ext, psi_ext, J_z_ext, Z_ext;
    
    // Metric coefficients
    Field2D Rxy, Bpxy, Btxy, hthe;
    
    const BoutReal MU0 = 4.0e-7 * PI;
    const BoutReal Charge = 1.60217646e-19;   // electron charge
    const BoutReal Mi = 2.0 * 1.67262158e-27; // Ion mass
    const BoutReal Me = 9.1093816e-31;        // Electron mass
    const BoutReal Me_Mi = Me / Mi;           // Electron mass / Ion mass

    // normalisation parameters
    BoutReal Tenorm, Nenorm, Bnorm;
    BoutReal Cs, rho_m, wci, R0;

    // collisional resistivity, viscosity and cross-field thermal transport coefficients
    BoutReal eta, mu, rho_perp;
    
    bool include_Jz0;
    int jpar_bndry;

    // Poisson brackets: b0 x Grad(f) dot Grad(g) / B = [f, g]
    // Method to use: BRACKET_ARAKAWA, BRACKET_STD or BRACKET_SIMPLE
    BRACKET_METHOD bm; // Bracket method for advection terms
    
    InvertPar *inv; // Parallel inversion class used in preconditioner

    // Coordinate systen netric
    Coordinates *coord;

    // Inverts a laplacian to get potential
    Laplacian *phiSolver;

    int init(bool restarting) override {
        
        coord = mesh->getCoordinates();
        GRID_LOAD(Rxy, Bpxy, Btxy, hthe);
        mesh->get(coord->Bxy, "Bxy");
        
        
        // Get the options
        auto& options = Options::root()["carreras"];
        
        eta = options["eta"].doc("Normalised resistivity").withDefault(1.0e-3);
        mu = options["mu"].doc("Normalised viscosity").withDefault(1.e-3);
        chi_perp = options["chi_perp"].doc("Cross-field transport coefficient").withDefault(1.e-3);

        // change later, not important right now
        bm = BRACKET_STD;
    
        include_Jz0 = options["include_Jz0"].withDefault(true);
        J_z_bndry = options["J_z_bndry"].withDefault(0.0);

        //////////////////////////////////////////////////
        // Normalisation
        
        Tenorm = 1000;

        Nenorm = 1.e19;

        Bnorm = max(coord->Bxy, true);

        Cs = sqrt(Charge * Tenorm / Mi);

        // drift scale
        rho_m = Cs * Mi / (Charge * Bnorm);

        R0 = MU0 * Charge * Tenorm * Nenrom / (Bnorm * Bnorm);

        output << "\tNormalisations:" << endl;
        output << "\tCs       : " << Cs << endl;
        output << "\trho_m    : " << rho_m << endl;
        output << "\twci      : " << wci << endl;
        output << "\tR0 : " << beta_hat << endl;

        SAVE_ONCE(Tenorm, Nenorm, Bnorm);
        SAVE_ONCE(Cs, rho_m, wci, R0);

        // Normalise geometry
        Rxy /= rho_m;
        hthe /= rho_m;
        coord->dx /= rho_m * rho_m * Bnorm;

        // Normalise magnetic field
        Bpxy /= Bnorm;
        Btxy /= Bnorm;
        coord->Bxy /= Bnorm;

        // Plasma quantities
        Jz0 /= Nenorm * Charge * Cs;

        // CALCULATE METRICS
        
        coord->g11 = SQ(Rxy * Bpxy);
        coord->g22 = 1.0 / SQ(hthe);
        coord->g33 = SQ(coord->Bxy) / coord->g11;
        coord->g12 = 0.0;
        coord->g13 = 0.;
        coord->g23 = -Btxy / (hthe * Bpxy * Rxy);

        coord->J = hthe / Bpxy;

        coord->g_11 = 1.0 / coord->g11;
        coord->g_22 = SQ(coord->Bxy * hthe / Bpxy);
        coord->g_33 = Rxy * Rxy;
        coord->g_12 = 0.0;
        coord->g_13 = 0.0;
        coord->g_23 = Btxy * hthe * Rxy / Bpxy;

        coord->geometry();

        // Tell BOUT++ which variables to evolve
        bout_solve(psi, "Poloidal Flux");
        bout_solve(U, "Vorticity");
        bout_sove(p_hat, "fluctuating pressure");

        // Set boundary conditions
        J_z.setBoundary("J_z");
        phi.setBoundary("phi");

        // Add any other variables to be dumped to file
        SAVE_REPEAT(phi, J_z);
        SAVE_ONCE(Jz0);

        // Generate external field
        initial_profile("");
    }

    int rhs(BoutReal UNUSED(time)) override {
        // invert laplace to obtain phi from U
        phi = invert_laplace(mesh->Bxy*U, phi_flags);
         
        // define reduced resistive MHD equations for Resistive Balooning Mode
        mesh->communicate(psi, U, phi);

        // calculate laplace to find J_z given psi;
        J_z = Delp2(psi) / (mu_0*R_0);
        
        // still need to figure out how to make this functional
        ddt(psi) = -R_0 * Grad_par(phi) + R_0 * eta * J_z;
        ddt(U) = -Grad_par(J_z) * V_dot_Grad(Z, Grad_cross_grad(omega, p_hat)) + mu*Delp2(U); // Assume Z is a vector
        ddt(p_hat) = -Grad_par(p_hat) * chi_perp - V * Rxy * Grad_perp(p_0);

    }
};
