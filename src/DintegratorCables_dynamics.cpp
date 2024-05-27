#include "DintegratorCables_dynamics.hpp"
#include <cmath>
// Auto generated file
// Created at: 2024-05-27--18-20-10

namespace dynobench {
void calcV_DintegratorCables(double* ff,
                double m0, double m1, double m2, double l1, double l2,
                const double *x, const double *u)
{

    ff[0] = x[4];
    ff[1] = x[5];
    ff[2] = x[6];
    ff[3] = x[7];
    ff[4] = (u[0] + u[2])/(m0 + m1 + m2);
    ff[5] = (u[1] + u[3])/(m0 + m1 + m2);
    ff[6] = (-l1*u[0]*sin(x[2]) - l2*u[2]*sin(x[3]))*1.0/(m1*pow(sin(x[2]), 2)*(l1*l1) + m1*pow(cos(x[2]), 2)*(l1*l1));
    ff[7] = (l1*u[1]*cos(x[2]) + l2*u[3]*cos(x[3]))*1.0/(m2*pow(sin(x[3]), 2)*(l2*l2) + m2*pow(cos(x[3]), 2)*(l2*l2));

}
void calcStep_DintegratorCables(double* xnext, 
        double m0, double m1, double m2, double l1, double l2, 
        const double *x, const double *u, double dt)
{

    xnext[0] = dt*x[4] + x[0];
    xnext[1] = dt*x[5] + x[1];
    xnext[2] = dt*x[6] + x[2];
    xnext[3] = dt*x[7] + x[3];
    xnext[4] = dt*(u[0] + u[2])/(m0 + m1 + m2) + x[4];
    xnext[5] = dt*(u[1] + u[3])/(m0 + m1 + m2) + x[5];
    xnext[6] = dt*(-l1*u[0]*sin(x[2]) - l2*u[2]*sin(x[3]))/(pow(l1, 2)*m1*pow(sin(x[2]), 2) + pow(l1, 2)*m1*pow(cos(x[2]), 2)) + x[6];
    xnext[7] = dt*(l1*u[1]*cos(x[2]) + l2*u[3]*cos(x[3]))/(pow(l2, 2)*m2*pow(sin(x[3]), 2) + pow(l2, 2)*m2*pow(cos(x[3]), 2)) + x[7];

}
void calcJ_DintegratorCables(
            double* Jx, 
            double* Ju, 
            double m0, double m1, double m2, double l1, double l2,
                const double *x, const double *u)
{
    Jx[32] = 1;
    Jx[41] = 1;
    Jx[50] = 1;
    Jx[59] = 1;
    Jx[22] = -l1*u[0]*cos(x[2])*1.0/(m1*pow(sin(x[2]), 2)*(l1*l1) + m1*pow(cos(x[2]), 2)*(l1*l1));
    Jx[30] = -l2*u[2]*cos(x[3])*1.0/(m1*pow(sin(x[2]), 2)*(l1*l1) + m1*pow(cos(x[2]), 2)*(l1*l1));
    Jx[23] = -l1*u[1]*sin(x[2])*1.0/(m2*pow(sin(x[3]), 2)*(l2*l2) + m2*pow(cos(x[3]), 2)*(l2*l2));
    Jx[31] = -l2*u[3]*sin(x[3])*1.0/(m2*pow(sin(x[3]), 2)*(l2*l2) + m2*pow(cos(x[3]), 2)*(l2*l2));
    Ju[4] = 1.0/(m0 + m1 + m2);
    Ju[20] = 1.0/(m0 + m1 + m2);
    Ju[13] = 1.0/(m0 + m1 + m2);
    Ju[29] = 1.0/(m0 + m1 + m2);
    Ju[6] = -l1*sin(x[2])*1.0/(m1*pow(sin(x[2]), 2)*(l1*l1) + m1*pow(cos(x[2]), 2)*(l1*l1));
    Ju[22] = -l2*sin(x[3])*1.0/(m1*pow(sin(x[2]), 2)*(l1*l1) + m1*pow(cos(x[2]), 2)*(l1*l1));
    Ju[15] = l1*cos(x[2])*1.0/(m2*pow(sin(x[3]), 2)*(l2*l2) + m2*pow(cos(x[3]), 2)*(l2*l2));
    Ju[31] = l2*cos(x[3])*1.0/(m2*pow(sin(x[3]), 2)*(l2*l2) + m2*pow(cos(x[3]), 2)*(l2*l2));

}
void calcF_DintegratorCables(
        double* Fx, 
        double* Fu, 
        double m0, double m1, double m2, double l1, double l2,
        const double *x, const double *u, double dt)
{
    Fx[0] = 1;
    Fx[32] = dt;
    Fx[9] = 1;
    Fx[41] = dt;
    Fx[18] = 1;
    Fx[50] = dt;
    Fx[27] = 1;
    Fx[59] = dt;
    Fx[36] = 1;
    Fx[45] = 1;
    Fx[22] = -dt*l1*u[0]*cos(x[2])*1.0/(m1*pow(sin(x[2]), 2)*(l1*l1) + m1*pow(cos(x[2]), 2)*(l1*l1));
    Fx[30] = -dt*l2*u[2]*cos(x[3])*1.0/(m1*pow(sin(x[2]), 2)*(l1*l1) + m1*pow(cos(x[2]), 2)*(l1*l1));
    Fx[54] = 1;
    Fx[23] = -dt*l1*u[1]*sin(x[2])*1.0/(m2*pow(sin(x[3]), 2)*(l2*l2) + m2*pow(cos(x[3]), 2)*(l2*l2));
    Fx[31] = -dt*l2*u[3]*sin(x[3])*1.0/(m2*pow(sin(x[3]), 2)*(l2*l2) + m2*pow(cos(x[3]), 2)*(l2*l2));
    Fx[63] = 1;
    Fu[4] = dt/(m0 + m1 + m2);
    Fu[20] = dt/(m0 + m1 + m2);
    Fu[13] = dt/(m0 + m1 + m2);
    Fu[29] = dt/(m0 + m1 + m2);
    Fu[6] = -dt*l1*sin(x[2])*1.0/(m1*pow(sin(x[2]), 2)*(l1*l1) + m1*pow(cos(x[2]), 2)*(l1*l1));
    Fu[22] = -dt*l2*sin(x[3])*1.0/(m1*pow(sin(x[2]), 2)*(l1*l1) + m1*pow(cos(x[2]), 2)*(l1*l1));
    Fu[15] = dt*l1*cos(x[2])*1.0/(m2*pow(sin(x[3]), 2)*(l2*l2) + m2*pow(cos(x[3]), 2)*(l2*l2));
    Fu[31] = dt*l2*cos(x[3])*1.0/(m2*pow(sin(x[3]), 2)*(l2*l2) + m2*pow(cos(x[3]), 2)*(l2*l2));

}

}
