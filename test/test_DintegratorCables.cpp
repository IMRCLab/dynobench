#include <boost/test/unit_test.hpp>

#include "dynobench/motions.hpp"
#include "dynobench/DintegratorCables.hpp"

#define base_path "../"


using namespace dynobench;

BOOST_AUTO_TEST_CASE(t_DintegratorCables) {

  dynobench::DintegratorCables_params params;
  params.read_from_yaml(base_path "models/DintegratorCables.yaml");
  auto model = mk<dynobench::DintegratorCables>(params);

}

BOOST_AUTO_TEST_CASE(t_DintegratorCables_dynamics) { 

  dynobench::DintegratorCables_params params;
  params.read_from_yaml(base_path "models/DintegratorCables.yaml");
  auto model = mk<dynobench::DintegratorCables>(params);

  Eigen::VectorXd x0 = Eigen::VectorXd::Zero(model->nx);

  int nx = model->nx;
  int nu = model->nu;

  Eigen::VectorXd x_default(nx), u_default(nu);
  x_default.setZero();
  
  std::map<std::string, std::vector<double>> info;
  info = model->get_info(x_default);
  

  std::cout<< "params m0: " << model->params.m0 << std::endl;
  std::cout<< "params m1: " << model->params.m1 << std::endl;
  std::cout<< "params m2: " << model->params.m2 << std::endl;
  std::cout<< "params l1: " << model->params.l1 << std::endl;
  std::cout<< "params l2: " << model->params.l2 << std::endl;
  // Printing each element of the map
  for (const auto& pair : info) {
      std::cout << pair.first << ": ";
      for (const double val : pair.second) {
        std::cout << val << " ";
      }
      std::cout << std::endl;
  }
  std::cout << "state: " << nx << std::endl;
  std::cout << "input: " << nu << std::endl;
  // exit(3);
  u_default = model->u_0;

  Eigen::VectorXd xrand(nx), urand(nu) , xrandnoise(nx) , urandnoise(nx);
  xrand.setZero(); // TODO: DONE
  xrand << 3., 2., 10., 11., 3., 4., 5., 6. ; 
  urand << 10.0, 2.0, 10.0, 2.0;


  xrandnoise = xrand + 0.01 * Eigen::VectorXd::Random(nx);
  model->ensure(xrandnoise);
  urandnoise = urand + 0.01 * Eigen::VectorXd::Random(nu);


  Eigen::MatrixXd Jx_diff(nx, nx), Ju_diff(nx, nu), Jx(nx, nx), Ju(nx, nu);
  Eigen::MatrixXd Sx_diff(nx, nx), Su_diff(nx, nu), Sx(nx, nx), Su(nx, nu);

  std::vector<std::pair<Eigen::VectorXd, Eigen::VectorXd>> xu_s;


  Eigen::VectorXd ff(nx);
  ff.setZero();
  xu_s.push_back({x_default, u_default});
  xu_s.push_back({xrand, urand});
  xu_s.push_back({xrandnoise, urandnoise});

  double dt = model->ref_dt;

  for (const auto &k : xu_s) {
    const auto &x0 = k.first;
    const auto &u0 = k.second;

    for (auto &m_ptr :
         {&Jx_diff, &Ju_diff, &Jx, &Ju, &Sx_diff, &Su_diff, &Sx, &Su}) {
      m_ptr->setZero();
    }

    CSTR_V(x0);
    CSTR_V(u0);

    model->calcDiffV(Jx, Ju, x0, u0);
    model->stepDiff(Sx, Su, x0, u0, dt);
    model->calcV(ff, x0, u0);

    std::cout << "ff: \n"<< ff << "\n\n" << std::endl;
    
    
    finite_diff_jac(
        [&](const Eigen::VectorXd &x, Eigen::Ref<Eigen::VectorXd> y) {
          model->calcV(y, x, u0);
        },
        x0, nx, Jx_diff);

    finite_diff_jac(
        [&](const Eigen::VectorXd &u, Eigen::Ref<Eigen::VectorXd> y) {
          model->calcV(y, x0, u);
        },
        u0, nx, Ju_diff);

    finite_diff_jac(
        [&](const Eigen::VectorXd &x, Eigen::Ref<Eigen::VectorXd> y) {
          model->step(y, x, u0, dt);
        },
        x0, nx, Sx_diff);

    finite_diff_jac(
        [&](const Eigen::VectorXd &u, Eigen::Ref<Eigen::VectorXd> y) {
          model->step(y, x0, u, dt);
        },
        u0, nx, Su_diff);
    std::cout << "Jx: \n" << Jx << std::endl;
    std::cout << "Jx_diff: \n" << Jx_diff << std::endl;

    std::cout << "-----------\n"
              << "report Jx " << std::endl;
    approx_equal_report(Jx, Jx_diff);
    std::cout << "report Ju " << std::endl;
    approx_equal_report(Ju, Ju_diff);

    std::cout << "report Sx " << std::endl;
    approx_equal_report(Sx, Sx_diff);
    std::cout << "report Su " << std::endl;
    approx_equal_report(Su, Su_diff);

    BOOST_TEST((Jx - Jx_diff).norm() <= 10 * 1e-4);
    BOOST_TEST((Ju - Ju_diff).norm() <= 10 * 1e-4);

    BOOST_TEST((Sx - Sx_diff).norm() <= 10 * 1e-4);
    BOOST_TEST((Su - Su_diff).norm() <= 10 * 1e-4);
  }
}