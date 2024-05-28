#include "dynobench/DintegratorCables.hpp"
#include "DintegratorCables_dynamics.hpp" // @KHALED TODO (e.g. n=1, point mass)
#include <fcl/broadphase/broadphase_dynamic_AABB_tree.h>
#include <fcl/broadphase/default_broadphase_callbacks.h>
#include <fcl/geometry/shape/box.h>
#include <fcl/geometry/shape/capsule.h>
#include <fcl/geometry/shape/sphere.h>
// #include "quadrotor_payload_dynamics_autogen_n3_p.hpp" // @KHALED TODO (e.g.
// n=2, point mass)

namespace dynobench {

void DintegratorCables_params::read_from_yaml(YAML::Node &node) {
  set_from_yaml(node, VAR_WITH_NAME(col_size_robot));
  set_from_yaml(node, VAR_WITH_NAME(col_size_payload));
  set_from_yaml(node, VAR_WITH_NAME(max_vel));
  set_from_yaml(node, VAR_WITH_NAME(max_acc));
  set_from_yaml(node, VAR_WITH_NAME(max_angular_vel));
  set_from_yaml(node, VAR_WITH_NAME(m0));
  set_from_yaml(node, VAR_WITH_NAME(m1));
  set_from_yaml(node, VAR_WITH_NAME(m2));
  set_from_yaml(node, VAR_WITH_NAME(l1));
  set_from_yaml(node, VAR_WITH_NAME(l2));
  set_from_yaml(node, VAR_WITH_NAME(distance_weights));
  set_from_yaml(node, VAR_WITH_NAME(size));
  set_from_yaml(node, VAR_WITH_NAME(u_ub));
  set_from_yaml(node, VAR_WITH_NAME(u_lb));
  set_from_yaml(node, VAR_WITH_NAME(dt));
}

void DintegratorCables_params::read_from_yaml(const char *file) {
  std::cout << "loading file: " << file << std::endl;
  filename = file;
  YAML::Node node = YAML::LoadFile(file);
  read_from_yaml(node);
}

DintegratorCables::DintegratorCables(const DintegratorCables_params &params, 
                                     const Eigen::Vector2d &p_lb,
                                     const Eigen::Vector2d &p_ub)
    : Model_robot(std::make_shared<Rn>(8), 4), params(params) {

  // description of state and control
  x_desc = {"p0x[m]", "p0y[m]", "th1[rad]", "th2[rad]", "dpx[m/s]", "dpy[m/s]", "dth1[rad/s]", "dth2[rad/s]",};
  u_desc = {"ddp1x[m/s^2]", "ddp1y[m/s^2]", "ddp2x[m/s^2]", "ddp2y[m/s^2]"};

  const double RM_max__ = std::sqrt(std::numeric_limits<double>::max());
  const double RM_low__ = -RM_max__;

  using V4d = Eigen::Vector4d;
  using Vxd = Eigen::VectorXd;

  std::cout << "Robot name " << name << std::endl;
  std::cout << "Parameters" << std::endl;
  this->params.write(std::cout);
  std::cout << "***" << std::endl;

  name = "DintegratorCables";
  translation_invariance = 2;
  invariance_reuse_col_shape = false;
  nx_col = 8; 
  nx_pr = 2;
  is_2d = true;

  ref_dt = params.dt;
  distance_weights = params.distance_weights;
  // bound on state and control
  u_lb << -params.max_acc, -params.max_acc, -params.max_acc, -params.max_acc;
  u_ub << params.max_acc, params.max_acc, params.max_acc, params.max_acc;

  x_lb << low__, low__, low__, low__, -params.max_vel, -params.max_vel, -params.max_angular_vel, -params.max_angular_vel;
  x_lb << max__, max__, max__, max__, -params.max_vel, -params.max_vel, params.max_angular_vel, params.max_angular_vel;

  // add bounds on position if provided
  if (p_lb.size() && p_ub.size()) {
    set_position_lb(p_lb);
    set_position_ub(p_ub);
  }

  // COLLISIONS
  collision_geometries.clear();

  double rate_colision_cables =
      .2; // we use a shorter collision body for the
          // cables to avoid self collision against payload or robot!
  collision_geometries.emplace_back(
      std::make_shared<fcl::Sphered>(params.col_size_payload));

  collision_geometries.emplace_back(std::make_shared<fcl::Capsuled>(
        params.col_size_payload, rate_colision_cables * params.l1));

  collision_geometries.emplace_back(std::make_shared<fcl::Capsuled>(
        params.col_size_payload, rate_colision_cables * params.l2));

  collision_geometries.emplace_back(
        std::make_shared<fcl::Sphered>(params.col_size_robot));

  collision_geometries.emplace_back(
        std::make_shared<fcl::Sphered>(params.col_size_robot));


  ts_data.resize(5);
  col_outs.resize(5);

  if (p_lb.size() && p_ub.size()) {
    set_position_lb(p_lb);
    set_position_ub(p_ub);
  }

  for (auto &c : collision_geometries) {
    collision_objects.emplace_back(std::make_unique<fcl::CollisionObjectd>(c));
  }
  col_mng_robots_ = std::make_shared<fcl::DynamicAABBTreeCollisionManagerd>();
  col_mng_robots_->setup();

}

std::map<std::string, std::vector<double>>
DintegratorCables::get_info(const Eigen::Ref<const Eigen::VectorXd> &x) {

  std::map<std::string, std::vector<double>> out;

  Eigen::VectorXd pr1(Eigen::Vector3d::Zero());
  Eigen::VectorXd pr2(Eigen::Vector3d::Zero());
  Eigen::VectorXd pc1(Eigen::Vector3d::Zero());
  Eigen::VectorXd pc2(Eigen::Vector3d::Zero());
  get_robot1_pos(x, pr1);
  get_cable1_center_pos(x, pc1);
  get_robot2_pos(x, pr2);
  get_cable2_center_pos(x, pc2);

  out.insert({"robot_pos1", {pr1(0), pr1(1), pr1(2)}});
  out.insert({"cable_pos1", {pc1(0), pc1(1), pc1(2)}});
  out.insert({"robot_pos2", {pr2(0), pr2(1), pr2(2)}});
  out.insert({"cable_pos2", {pc2(0), pc2(1), pc2(2)}});
  return out;
}

void DintegratorCables::transformation_collision_geometries(
  const Eigen::Ref<const Eigen::VectorXd> &x, std::vector<Transform3d> &ts) {
    // shapes:
    // payload, robot 1, robot 2 cable 1, cable 2

    //payload
    Eigen::Vector2d pos_p;
    get_payload_pos(x, pos_p);

    fcl::Transform3d result_p;
    result_p = Eigen::Translation<double, 3>(pos_p(0), pos_p(1), 0.0);
    result_p.rotate(
      Eigen::AngleAxisd(0., Eigen::Vector3d::UnitZ())
    );
    ts.at(0) = result_p;

    // robot 1 and 2 
    Eigen::Vector2d robot1;
    Eigen::Vector2d robot2;
    get_robot1_pos(x, robot1);
    get_robot2_pos(x, robot2);

    fcl::Transform3d result_r1;
    fcl::Transform3d result_r2;

    result_r1 = Eigen::Translation<double, 3>(robot1(0), robot1(1), 0.0);
    result_r1.rotate(
      Eigen::AngleAxisd(0., Eigen::Vector3d::UnitZ())
    );
    ts.at(1) = result_r1;

    result_r2 = Eigen::Translation<double, 3>(robot2(0), robot2(1), 0.0);
    result_r2.rotate(
      Eigen::AngleAxisd(0., Eigen::Vector3d::UnitZ())
    );
    ts.at(2) = result_r2;


    // cable 1 and 2 
    Eigen::Vector2d cable1;
    Eigen::Vector2d cable2;
    double th1;
    double th2;
    
    get_cable1_center_pos(x, cable1);
    get_cable2_center_pos(x, cable2);
    get_th1(x, th1);
    get_th2(x, th2);

    fcl::Transform3d result_c1;
    fcl::Transform3d result_c2;

    result_c1 = Eigen::Translation<double, 3>(cable1(0), cable1(1), 0.0);
    result_c1.rotate(Eigen::AngleAxisd(th1, Eigen::Vector3d::UnitZ()));
    ts.at(3) = result_c1;

    result_c2 = Eigen::Translation<double, 3>(cable2(0), cable2(1), 0.0);
    result_c2.rotate(
      Eigen::AngleAxisd(th2, Eigen::Vector3d::UnitZ())
    );
    ts.at(4) = result_c2;
  }

void DintegratorCables::calcV(Eigen::Ref<Eigen::VectorXd> ff,
                                  const Eigen::Ref<const Eigen::VectorXd> &x,
                                  const Eigen::Ref<const Eigen::VectorXd> &u) {

  // Call a function in the autogenerated file

  auto apply_fun = [&](auto &fun) {
    fun(ff.data(), params.m0, params.m1, params.m2,
        params.l1, params.l2, x.data(), u.data());
  };
  apply_fun(calcV_DintegratorCables);

}

void DintegratorCables::calcDiffV(
    Eigen::Ref<Eigen::MatrixXd> Jv_x, Eigen::Ref<Eigen::MatrixXd> Jv_u,
    const Eigen::Ref<const Eigen::VectorXd> &x,
    const Eigen::Ref<const Eigen::VectorXd> &u) {

  // Call a function in the autogenerated file
  // NOT_IMPLEMENTED_TODO;

  auto apply_fun = [&](auto &fun) {
    fun(Jv_x.data(), Jv_u.data(), params.m0, params.m1, params.m2,
        params.l1, params.l2, x.data(), u.data());
  };

  apply_fun(calcJ_DintegratorCables);

}

void DintegratorCables::step(Eigen::Ref<Eigen::VectorXd> xnext,
                                 const Eigen::Ref<const Eigen::VectorXd> &x,
                                 const Eigen::Ref<const Eigen::VectorXd> &u,
                                 double dt) {

  // Call a function in the autogenerated file
  auto apply_fun = [&](auto &fun) {
    fun(xnext.data(), params.m0, params.m1, params.m2,
        params.l1, params.l2, x.data(), u.data(), dt);
  };
  apply_fun(calcStep_DintegratorCables);
  ensure(xnext);
}

void DintegratorCables::stepDiff(Eigen::Ref<Eigen::MatrixXd> Fx,
                                     Eigen::Ref<Eigen::MatrixXd> Fu,
                                     const Eigen::Ref<const Eigen::VectorXd> &x,
                                     const Eigen::Ref<const Eigen::VectorXd> &u,
                                     double dt) {

  // Call a function in the autogenerated file
  auto apply_fun = [&](auto &fun) {
    fun(Fx.data(), Fu.data(), params.m0, params.m1, params.m2,
        params.l1, params.l2, x.data(), u.data(), dt);
  };

  // TODO: check if this is required!
  Fx.setZero();
  Fu.setZero();
  apply_fun(calcF_DintegratorCables);

}

} // namespace dynobench
