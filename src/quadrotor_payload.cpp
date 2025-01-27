#include "dynobench/quadrotor_payload.hpp"
#include "quadrotor_payload_dynamics_autogen.hpp"
#include <fcl/geometry/shape/box.h>
#include <fcl/geometry/shape/sphere.h>

namespace dynobench {

void Quad3dpayload_params::read_from_yaml(YAML::Node &node) {

  set_from_yaml(node, VAR_WITH_NAME(m_payload));
  set_from_yaml(node, VAR_WITH_NAME(l_payload));

  set_from_yaml(node, VAR_WITH_NAME(max_vel));
  set_from_yaml(node, VAR_WITH_NAME(max_angular_vel));
  set_from_yaml(node, VAR_WITH_NAME(max_acc));
  set_from_yaml(node, VAR_WITH_NAME(max_angular_acc));
  set_from_yaml(node, VAR_WITH_NAME(motor_control));
  set_from_yaml(node, VAR_WITH_NAME(m));
  set_from_yaml(node, VAR_WITH_NAME(g));
  set_from_yaml(node, VAR_WITH_NAME(max_f));
  set_from_yaml(node, VAR_WITH_NAME(arm_length));
  set_from_yaml(node, VAR_WITH_NAME(t2t));
  set_from_yaml(node, VAR_WITH_NAME(dt));
  set_from_yaml(node, VAR_WITH_NAME(shape));
  set_from_yaml(node, VAR_WITH_NAME(J_v));
  set_from_yaml(node, VAR_WITH_NAME(distance_weights));
  set_from_yaml(node, VAR_WITH_NAME(u_ub));
  set_from_yaml(node, VAR_WITH_NAME(u_lb));

  set_from_yaml(node, VAR_WITH_NAME(size));
}

void Quad3dpayload_params::read_from_yaml(const char *file) {
  std::cout << "loading file: " << file << std::endl;
  filename = file;
  YAML::Node node = YAML::LoadFile(file);
  read_from_yaml(node);
}

Model_quad3dpayload::Model_quad3dpayload(const Quad3dpayload_params &params,

                                         const Eigen::VectorXd &p_lb,
                                         const Eigen::VectorXd &p_ub)

    : Model_robot(std::make_shared<Rn>(13), 4),
      params(params) // TODO: KHALED adjust to nx and nu that you want to use
{

  // TODO: KHALED adapt this

  const double RM_max__ = std::sqrt(std::numeric_limits<double>::max());
  const double RM_low__ = -RM_max__;

  using V4d = Eigen::Vector4d;
  using Vxd = Eigen::VectorXd;

  std::cout << "Robot name " << name << std::endl;
  std::cout << "Parameters" << std::endl;
  this->params.write(std::cout);
  std::cout << "***" << std::endl;

  if (params.motor_control) {
    u_0.setOnes();
  } else {
    u_0 << 1, 0, 0, 0;
  }

  translation_invariance = 3;
  invariance_reuse_col_shape = false;
  nx_col = 7;
  nx_pr = 7;
  is_2d = false;

  ref_dt = params.dt;
  distance_weights = params.distance_weights;

  arm = 0.707106781 * params.arm_length;
  u_nominal = params.m * g / 4.;

  if (params.motor_control) {
    B0 << 1, 1, 1, 1, -arm, -arm, arm, arm, -arm, arm, arm, -arm, -params.t2t,
        params.t2t, -params.t2t, params.t2t;
    B0 *= u_nominal;
    B0inv = B0.inverse();
  } else {
    B0.setIdentity();
    double nominal_angular_acceleration = 20;
    B0(0, 0) *= u_nominal * 4;
    B0(1, 1) *= nominal_angular_acceleration;
    B0(2, 2) *= nominal_angular_acceleration;
    B0(3, 3) *= nominal_angular_acceleration;
  }

  name = "quad3dpayload";
  x_desc = {"x [m]",      "y [m]",      "z [m]",     "qx []",    "qy []",
            "qz []",      "qw []",      "vx [m/s]",  "vy [m/s]", "vz [m/s]",
            "wx [rad/s]", "wy [rad/s]", "wz [rad/s]"}; // TODO: Khaled adjust

  u_desc = {"f1 []", "f2 [], f3 [], f4 []"};

  Fu_selection.setZero();
  Fu_selection(2, 0) = 1.;

  // [ 0, 0, 0, 0]   [eta(0)]    =
  // [ 0, 0, 0, 0]   [eta(1)]
  // [ 1, 0, 0, 0]   [eta(2)]
  //                 [eta(3)]

  Ftau_selection.setZero();
  Ftau_selection(0, 1) = 1.;
  Ftau_selection(1, 2) = 1.;
  Ftau_selection(2, 3) = 1.;

  // [ 0, 1, 0, 0]   [eta(0)]    =
  // [ 0, 0, 1, 0]   [eta(1)]
  // [ 0, 0, 0, 1]   [eta(2)]
  //                 [eta(3)]

  Fu_selection_B0 = Fu_selection * B0;
  Ftau_selection_B0 = Ftau_selection * B0;

  // Bounds

  if (params.motor_control) {
    u_lb = Eigen::Vector4d(0, 0, 0, 0);
    u_ub =
        Eigen::Vector4d(params.max_f, params.max_f, params.max_f, params.max_f);
  } else {
    u_lb = params.u_lb;
    u_ub = params.u_ub;
  }

  // TODO: Khaled ajust bounds in the state space
  x_lb.segment(0, 7) << RM_low__, RM_low__, RM_low__, RM_low__, RM_low__,
      RM_low__, RM_low__;
  x_lb.segment(7, 3) << -params.max_vel, -params.max_vel, -params.max_vel;
  x_lb.segment(10, 3) << -params.max_angular_vel, -params.max_angular_vel,
      -params.max_angular_vel;

  x_ub.segment(0, 7) << RM_max__, RM_max__, RM_max__, RM_max__, RM_max__,
      RM_max__, RM_max__;
  x_ub.segment(7, 3) << params.max_vel, params.max_vel, params.max_vel;
  x_ub.segment(10, 3) << params.max_angular_vel, params.max_angular_vel,
      params.max_angular_vel;

  // some precomputation
  inverseJ_v = params.J_v.cwiseInverse();

  inverseJ_M = inverseJ_v.asDiagonal();
  J_M = params.J_v.asDiagonal();

  inverseJ_skew = Skew(inverseJ_v);
  J_skew = Skew(params.J_v);

  m_inv = 1. / params.m;
  m = params.m;
  grav_v = Eigen::Vector3d(0, 0, -params.m * g);

  u_weight = V4d(.5, .5, .5, .5);
  x_weightb = 50. * Vxd::Ones(13);
  x_weightb.head(7) = Eigen::VectorXd::Zero(7);

  if (params.shape == "box") {
    collision_geometries.emplace_back(std::make_shared<fcl::Boxd>(
        params.size(0), params.size(1), params.size(2)));
  } else if (params.shape == "sphere") {
    collision_geometries.emplace_back(
        std::make_shared<fcl::Sphered>(params.size(0)));
  } else {
    ERROR_WITH_INFO("not implemented");
  }

  if (p_lb.size() && p_ub.size()) {
    // TODO: Khalded adjust bounds --> maybe infinite it X quadrotor is not part
    // of the state
    set_position_lb(p_lb);
    set_position_ub(p_ub);
  }

  __Jv_x.resize(12, 13); // KHALED ajust
  __Jv_u.resize(12, 4);  // KHALED ajust

  __Jv_x.setZero(); // KHALED adjust
  __Jv_u.setZero(); // KHALED ajust
}

Eigen::VectorXd Model_quad3dpayload::get_x0(const Eigen::VectorXd &x) {
  // KHALED which is the state space?
  NOT_IMPLEMENTED_TODO;
}

void Model_quad3dpayload::sample_uniform(Eigen::Ref<Eigen::VectorXd> x) {
  NOT_IMPLEMENTED;
  // (void)x;
  // x = x_lb + (x_ub - x_lb)
  //                .cwiseProduct(.5 * (Eigen::VectorXd::Random(nx) +
  //                                    Eigen::VectorXd::Ones(nx)));
  // x.segment(3, 4) = Eigen::Quaterniond::UnitRandom().coeffs();
}

void Model_quad3dpayload::transformation_collision_geometries(
    const Eigen::Ref<const Eigen::VectorXd> &x, std::vector<Transform3d> &ts) {
  NOT_IMPLEMENTED_TODO;
}

void Model_quad3dpayload::collision_distance(
    const Eigen::Ref<const Eigen::VectorXd> &x, CollisionOut &cout) {
  NOT_IMPLEMENTED_TODO;
}

void Model_quad3dpayload::transform_primitive(
    const Eigen::Ref<const Eigen::VectorXd> &p,
    const std::vector<Eigen::VectorXd> &xs_in,
    const std::vector<Eigen::VectorXd> &us_in,
    std::vector<Eigen::VectorXd> &xs_out,
    std::vector<Eigen::VectorXd> &us_out) {
  NOT_IMPLEMENTED;
}

void Model_quad3dpayload::calcV(Eigen::Ref<Eigen::VectorXd> ff,
                                const Eigen::Ref<const Eigen::VectorXd> &x,
                                const Eigen::Ref<const Eigen::VectorXd> &u) {

  // Call a function in the autogenerated file
  NOT_IMPLEMENTED_TODO;
}

void Model_quad3dpayload::calcDiffV(
    Eigen::Ref<Eigen::MatrixXd> Jv_x, Eigen::Ref<Eigen::MatrixXd> Jv_u,
    const Eigen::Ref<const Eigen::VectorXd> &x,
    const Eigen::Ref<const Eigen::VectorXd> &u) {

  // Call a function in the autogenerated file
  NOT_IMPLEMENTED_TODO;
}

void Model_quad3dpayload::step(Eigen::Ref<Eigen::VectorXd> xnext,
                               const Eigen::Ref<const Eigen::VectorXd> &x,
                               const Eigen::Ref<const Eigen::VectorXd> &u,
                               double dt) {

  // Call a function in the autogenerated file
  NOT_IMPLEMENTED_TODO;
}

void Model_quad3dpayload::stepDiff(Eigen::Ref<Eigen::MatrixXd> Fx,
                                   Eigen::Ref<Eigen::MatrixXd> Fu,
                                   const Eigen::Ref<const Eigen::VectorXd> &x,
                                   const Eigen::Ref<const Eigen::VectorXd> &u,
                                   double dt) {

  // Call a function in the autogenerated file
  NOT_IMPLEMENTED_TODO;
}

double
Model_quad3dpayload::distance(const Eigen::Ref<const Eigen::VectorXd> &x,
                              const Eigen::Ref<const Eigen::VectorXd> &y) {
  NOT_IMPLEMENTED;
}

void Model_quad3dpayload::interpolate(
    Eigen::Ref<Eigen::VectorXd> xt,
    const Eigen::Ref<const Eigen::VectorXd> &from,
    const Eigen::Ref<const Eigen::VectorXd> &to, double dt) {
  NOT_IMPLEMENTED;
}

double

Model_quad3dpayload::lower_bound_time(
    const Eigen::Ref<const Eigen::VectorXd> &x,
    const Eigen::Ref<const Eigen::VectorXd> &y) {

  NOT_IMPLEMENTED;
}

double Model_quad3dpayload::lower_bound_time_pr(
    const Eigen::Ref<const Eigen::VectorXd> &x,
    const Eigen::Ref<const Eigen::VectorXd> &y) {

  NOT_IMPLEMENTED;
}

double Model_quad3dpayload::lower_bound_time_vel(
    const Eigen::Ref<const Eigen::VectorXd> &x,
    const Eigen::Ref<const Eigen::VectorXd> &y) {

  NOT_IMPLEMENTED;
  std::array<double, 2> maxs = {
      (x.segment<3>(7) - y.segment<3>(7)).norm() / params.max_acc,
      (x.segment<3>(10) - y.segment<3>(10)).norm() / params.max_angular_acc};

  return *std::max_element(maxs.cbegin(), maxs.cend());
}

} // namespace dynobench
