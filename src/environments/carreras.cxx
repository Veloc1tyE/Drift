// TODO: integrate physicsmodel solvers into environment

#include <bout/physicsmodel.hxx>

class ResistiveMHD : public PhysicsModel {
    // 3D evolving variables
    Vector3D phi; // velocity stream function
    Field3D p;
    
    // parameters
    BoutReal eta, mu, chi; // collisional resistivity, viscosity, and cross-field thermal transport coefficicients
    BoutReal mu_0; // vacuum permeability
    
    int init(bool restarting) override {
        // define variables
    }

    int rhs(BoutReal UNUSED(time)) override {
        // define reduced resistive MHD equations for Resistive Balooning Mode
    }


};
