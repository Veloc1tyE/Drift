// TODO: integrate physicsmodel solvers into environment
// TODO: Complete the simulation environment

#include <bout/physicsmodel.hxx>

class ResistiveMHD : public PhysicsModel {
    // 3D evolving variables
    Vector3D psi;
    Vector3D U, phi, J_z;
    Vector3D Z;
    
    Field3D p_hat; // fluctuating pressure field
    Field3D p; // time-averaged pressure field

    BoutReal eta, mu, chi; // collisional resistivity, viscosity, and cross-field thermal transport coefficicients
    BoutReal mu_0; // vacuum permeability
    BoutReal R_0; // Initial resisitivity
    BoutReal rho_m; // constant mass density
    BoutReal chi_perp; // convective transport coefficient

    // Coordinate system metric
    Coordinates *coord;

    // inversion
    InvertPar *inv;

    // Inverts a laplacian to get potential
    Laplacian *phiSolver;

    int init(bool restarting) override {
        // Get the options
        auto& options = Options::root()["carreras"];

        // Read from BOUT.inp, setting default to 1.0
        // The doc() provides some documentation in BOUT.settings
        eta = options["eta"].doc("collisional resistivity").withDefault(1.0);
        mu = options["mu"].doc("viscosity").withDefault(-1.0);
        chi = options["chi"].doc("cross-field thermal transport").withDefault(1.0);
        mu_0 = options["mu_0"].doc("vacuum permeability").withDefault(1.0);
        R_0 = options["R_0"].doc("Resistivity coefficient").withDefault(1.0);
        rho_m = options["rho_m"].doc("constant mass density").withDefault(1.0);
        
        psi.covariant = false;
        bout_solve(psi, "Poloidal Flux");
        bout_solve(U, "Vorticity");
        p.covariant = true;
        bout_sove(p_hat, "fluctuating pressure");
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
        ddt(p_hat) = -Grad_par(chi_perp) * p_hat - V_r * Grad_perp(p_0);

    }
};
