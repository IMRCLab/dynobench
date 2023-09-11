#include <fstream>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>

#include <Eigen/Dense>

#include "dynobench/croco_macros.hpp"
#include "dynobench/math_utils.hpp"
// #include <Eigen/src/plugins/BlockMethods.h>
#define BOOST_TEST_MODULE test_payload
#define BOOST_TEST_DYN_LINK
// #include "dynobench/motions.hpp"
// #include "dynobench/quadrotor_payload.hpp"
// #include "dynobench/quadrotor_payload_n.hpp"
#include <boost/test/unit_test.hpp>

#define base_path "../../"

using namespace dynobench;

using UAVParams = std::map<std::string, double>;
Eigen::IOFormat OctaveFmt(Eigen::StreamPrecision, 0, ", ", ";\n", "", "", "[",
                          "]");

Eigen::Vector4d intergrate_quat_formatC(const Eigen::Vector4d &__quat_p,
                                        const Eigen::Vector3d &wp, double dt) {

  Eigen::Vector4d quat_p(__quat_p(1), __quat_p(2), __quat_p(3), __quat_p(0));
  quat_p.normalize();
  Eigen::Vector4d __out, __out2;
  Eigen::Vector4d deltaQ;
  __get_quat_from_ang_vel_time(wp * dt, deltaQ, nullptr);
  quat_product(quat_p, deltaQ, __out, nullptr, nullptr);
  Eigen::Vector4d out(__out(3), __out(0), __out(1), __out(2));
  return out;
}

Eigen::Vector3d rotate_format_C(const Eigen::Vector4d &quat,
                                const Eigen::Vector3d &v) {

  Eigen::Vector4d q = quat.normalized();
  Eigen::Matrix3d R =
      Eigen::Quaterniond(q(0), q(1), q(2), q(3)).toRotationMatrix();
  return R * v;
}

// Assuming you have a function to_matrix implemented elsewhere, it would look
// something like:
Eigen::Matrix3d to_matrix_formatC(const Eigen::Vector4d &vec) {
  // TODO: check the order of the quaternion!!
  return Eigen::Quaterniond(vec(0), vec(1), vec(2), vec(3)).toRotationMatrix();
}

Eigen::Matrix3d skew(const Eigen::Vector3d &vec) {
  Eigen::Matrix3d skewMat;
  skewMat << 0, -vec(2), vec(1), vec(2), 0, -vec(0), -vec(1), vec(0), 0;
  return skewMat;
}

std::vector<std::vector<double>> readCSV(const std::string &filename) {
  std::ifstream file(filename);
  std::string line;
  std::vector<std::vector<double>> data;

  if (!file.is_open()) {
    std::cerr << "Failed to open the file: " << filename << std::endl;
    return data;
  }

  while (std::getline(file, line)) {
    std::stringstream lineStream(line);
    std::string cell;
    std::vector<double> rowData;

    while (std::getline(lineStream, cell, ',')) {
      rowData.push_back(stod(cell)); // Converts string to double
    }

    data.push_back(rowData);
  }

  file.close();
  return data;
}

struct UAV {

  Eigen::Matrix3d I =
      Eigen::Vector3d(16.571710e-6, 16.655602e-6, 29.261652e-6).asDiagonal();
  Eigen::Matrix3d invI = I.inverse();
  // Eigen::VectorXd states;
  double m = 0.0356;
  double l_c = .5;
  Eigen::Vector3d pos_fr_payload = Eigen::Vector3d(0., 0, 0.);

  void check() {
    // CHECK_EQ(states.size(), 7, "");
    CHECK((I.sum() > 0), "");
    CHECK((invI.sum() > 0), "");
    CHECK((pos_fr_payload.sum() > 0), "");
    CHECK((pos_fr_payload.sum() > 1e-6), "");
  }

  void getNextAngularState(Eigen::Ref<Eigen::Vector4d> next_q,
                           Eigen::Ref<Eigen::Vector3d> next_w,
                           const Eigen::Vector3d &curr_w,
                           const Eigen::Vector4d &curr_q,
                           const Eigen::Vector3d &tau, double dt) {

    Eigen::Matrix3d skew_matrix = skew(curr_w);
    Eigen::Vector3d wdot = invI * (tau - skew_matrix * I * curr_w);
    next_w = wdot * dt + curr_w;
    next_q = intergrate_quat_formatC(curr_q, curr_w, dt);
  }
};

class PayloadSystem {
public:
  PayloadSystem(const std::vector<UAV> &uavs) : uavs(uavs) {
    numOfquads = uavs.size();
    nx = 13 + numOfquads * 6 + numOfquads * 7;
    mt = numOfquads * 0.0356 + 0.0115; // TODO: this is not general
    payload_nx = 13;
    dt = 0.001;
    payload_nv = 6;
    payload_w_cables_nv = payload_nv + 3 * numOfquads;
    payload_w_cables_nx = payload_nx + 6 * numOfquads;
    accl.resize(payload_w_cables_nv);
    accl.setZero();
    J = Eigen::Vector3d(3.0002e-4, 7.7083e-7, 3.0075e-4).asDiagonal();
  }

  // Eigen::VectorXd plstate;
  int payload_w_cables_nv;

  int nx;
  int payload_nv;
  int payload_nx;
  int payload_w_cables_nx;
  bool pointmass = false;
  double mt; // total mass of system
  // Eigen::VectorXd state;
  Eigen::Matrix3d J;
  double dt;
  Eigen::VectorXd accl;
  int numOfquads;

  std::vector<UAV> uavs;
  // std::vector<Eigen::VectorXd> uav_states;

  // Eigen::VectorBlock<Eigen::VectorXd>
  auto get_state_uav_i(int i, Eigen::Ref<Eigen::VectorXd> state) {
    CHECK_EQ(state.size(), nx, "");
    CHECK_GEQ(i, 0, "");
    CHECK_GEQ(numOfquads, i + 1, "");
    return state.segment(payload_w_cables_nx + i * 7, 7);
  }

  auto get_state_uav_i_const(int i,
                             const Eigen::Ref<const Eigen::VectorXd> &state) {
    CHECK_EQ(state.size(), nx, "");
    CHECK_GEQ(i, 0, "");
    CHECK_GEQ(numOfquads, i + 1, "");
    return state.segment(payload_w_cables_nx + i * 7, 7);
  }

  // Eigen::VectorBlock<const Eigen::VectorXd> get_state_uav_i(int i, const
  // Eigen::VectorXd &state) {
  //   CHECK_EQ(state.size(), payload_w_cables_nv + 7 * numOfquads, "");
  //   CHECK_GEQ(i, 0, "");
  //   CHECK_GEQ(numOfquads, i + 1, "");
  //   return state.segment(payload_w_cables_nv + i * 7, 7);
  // }

  void check() { CHECK_EQ(uavs.size(), numOfquads, ""); }

  void getBq(Eigen::Ref<Eigen::MatrixXd> Bq,
             const Eigen::Ref<const Eigen::VectorXd> state) {
    // Assuming payload_w_cables_nv, payload_nv, payload_nx, mt, J, state, and
    // pointmass are member variables of the class Eigen::MatrixXd
    // Bq(payload_w_cables_nv, payload_w_cables_nv);

    CHECK_EQ(state.size(), nx, "");
    CHECK_EQ(Bq.rows(), payload_w_cables_nv, "");
    CHECK_EQ(Bq.cols(), payload_w_cables_nv, "");

    Bq.setZero();

    Bq.block<3, 3>(0, 0) = mt * Eigen::Matrix3d::Identity();

    std::cout << "state.transpose().format(OctaveFmt)" << std::endl;
    std::cout << state.transpose().format(OctaveFmt) << std::endl;

    int i = payload_nv;
    int k = payload_nx;

    for (int ii = 0; ii < numOfquads; ii++) {

      std::cout << " i "
                << " k " << i << " " << k << std::endl;

      double m = uavs.at(ii).m;
      double l = uavs.at(ii).l_c;

      Eigen::Vector3d qi = state.segment(k, 3);
      k += 3;

      Bq.block<3, 3>(i, 0) =
          -m * skew(qi); // Assuming skew() is defined elsewhere
      Bq.block<3, 3>(i, i) = m * l * Eigen::Matrix3d::Identity();

      if (!pointmass) {
        Eigen::Matrix3d R_p = to_matrix_formatC(
            state.segment(6, 4)); // Assuming to_matrix() is defined elsewhere
        Eigen::Vector3d posFrload(uavs.at(ii).pos_fr_payload);
        std::cout << "posFrload.format(OctaveFmt)" << std::endl;
        std::cout << posFrload.format(OctaveFmt) << std::endl;

        Eigen::Matrix3d qiqiT = qi * qi.transpose();

        Bq.block<3, 3>(0, 3) += -m * qiqiT * R_p * skew(posFrload);
        Bq.block<3, 3>(3, 0) += m * skew(posFrload) * R_p.transpose() * qiqiT;
        Bq.block<3, 3>(3, 3) += m * skew(posFrload) * R_p.transpose() * qiqiT *
                                R_p * skew(posFrload);
        Bq.block<3, 3>(i, 3) = m * skew(qi) * R_p * skew(posFrload);
        i += 3;
      }
    }

    if (!pointmass) {
      Bq.block<3, 3>(3, 3) = J - Bq.block<3, 3>(3, 3);
    }
  }

  void getNq(Eigen::Ref<Eigen::VectorXd> Nq,
             const Eigen::Ref<const Eigen::VectorXd> state) {
    // Assuming payload_w_cables_nv, payload_nv, payload_nx, mt, state are
    // member variables of the class Eigen::VectorXd Nq(payload_w_cables_nv);

    CHECK_EQ(Nq.size(), payload_w_cables_nv, "");
    CHECK_EQ(state.size(), nx, "");

    Nq.setZero();

    int i = payload_nv;
    int k = payload_nx;

    Eigen::Matrix3d R_p;
    Eigen::Vector3d wl;

    if (!pointmass) {
      R_p = to_matrix_formatC(
          state.segment(6, 4)); // Assuming to_matrix() is defined elsewhere
      wl = state.segment(10, 3);
    }

    // for (const auto &[name, uav] : uavs_params) {

    for (size_t ii = 0; ii < numOfquads; ii++) {

      double m = uavs.at(ii).m;
      double l = uavs.at(ii).l_c;

      Eigen::Vector3d qi = state.segment(k, 3);
      Eigen::Vector3d wi = state.segment(k + 3 * numOfquads, 3);

      k += 3;

      if (pointmass) {
        Nq.segment(0, 3) +=
            -m * l * wi.dot(wi) * qi; // Using Eigen's dot() for dot product
      } else {
        Eigen::Vector3d posFrload = uavs.at(ii).pos_fr_payload;
        Eigen::Matrix3d qiqiT = qi * qi.transpose();

        Nq.segment(0, 3) += (-m * l * wi.dot(wi) * qi -
                             m * qiqiT * R_p * skew(wl) * skew(wl) * posFrload);
        Nq.segment(3, 3) +=
            skew(posFrload) * R_p.transpose() *
            ((-m * l * wi.dot(wi) * qi) -
             (m * qiqiT * R_p * skew(wl) * skew(wl) * posFrload));
        Nq.segment(i, 3) = m * skew(qi) * R_p * skew(wl) * skew(wl) * posFrload;
        i += 3;
      }
    }

    if (!pointmass) {
      // Assuming J is a member variable representing inertia tensor
      Nq.segment(3, 3) -= skew(wl) * J * wl;
    }
  }

  void getuinp(Eigen::Ref<Eigen::VectorXd> u_inp,
               const Eigen::MatrixXd &ctrlInputs,
               const Eigen::Ref<const Eigen::VectorXd> state) {

    CHECK_EQ(u_inp.size(), payload_w_cables_nv, "");

    int i = 0, j = payload_nv, k = payload_nx;

    for (size_t ii = 0; ii < numOfquads; ++ii) {

      // const std::string &name = paramPair.first;
      // const UAVParams &uav = paramPair.second;

      double f = ctrlInputs(ii, 0);

      // std::cout << "uavs.at(name).state.segment(6,
      // 4).transpose().format(OctaveFmt)" << std::endl; std::cout <<
      // uavs.at(name).state.segment(6, 4).transpose().format(OctaveFmt) <<
      // std::endl;

      Eigen::Vector3d u_i =
          rotate_format_C(get_state_uav_i_const(ii, state).segment<4>(0),
                          Eigen::Vector3d(0, 0, f));
      std::cout << "u_i.format(OctaveFmt)" << std::endl;
      std::cout << u_i.format(OctaveFmt) << std::endl;

      Eigen::Vector3d posFrload;
      Eigen::Matrix3d R_p;
      Eigen::Vector3d wl;
      if (!pointmass) {
        R_p = to_matrix_formatC(state.segment(6, 4));
        wl = state.segment(10, 3);
        posFrload = uavs.at(ii).pos_fr_payload;
        // posFrload << uav.at("pos_fr_payloadx"), uav.at("pos_fr_payloady"),
        //     uav.at("pos_fr_payloadz");
      }

      Eigen::Vector3d qi = state.segment(k, 3);
      k += 3;

      Eigen::Matrix3d qiqiT = qi * qi.transpose();

      u_inp.segment(0, 3) += u_i;

      // Eigen::Vector3d u_perp = (Eigen::Matrix3d::Identity() - qiqiT) * u_i;
      u_inp.segment(j, 3) = -skew(qi) * u_i;

      if (!pointmass) {
        u_inp.segment(3, 3) += skew(posFrload) * R_p.transpose() * qiqiT * u_i;
      }

      i++;
      j += 3;
    }
  }

  // std::pair<std::map<std::string, UavModel>, Eigen::VectorXd>
  void stateEvolution(Eigen::Ref<Eigen::VectorXd> next_state,
                      const Eigen::Ref<const Eigen::VectorXd> &state,
                      const Eigen::MatrixXd &ctrlInputs) {

    CHECK_EQ(state.size(), nx, "");
    CHECK_EQ(next_state.size(), nx, "");

    Eigen::MatrixXd Bq(payload_w_cables_nv, payload_w_cables_nv);
    Eigen::VectorXd Nq(payload_w_cables_nv);
    Eigen::VectorXd u_inp(payload_w_cables_nv);

    Bq.setZero();
    Nq.setZero();
    u_inp.setZero();
    getBq(Bq, state);
    getNq(Nq, state);
    getuinp(u_inp, ctrlInputs, state);

    std::cout << "ctrlInputs.format(OctaveFmt)" << std::endl;
    std::cout << ctrlInputs.format(OctaveFmt) << std::endl;

    std::cout << "u_inp.format(OctaveFmt)" << std::endl;
    std::cout << u_inp.transpose().format(OctaveFmt) << std::endl;

    std::cout << "Bq " << std::endl;
    std::cout << Bq.format(OctaveFmt) << std::endl;
    std::cout << "Nq " << std::endl;
    std::cout << Nq.format(OctaveFmt) << std::endl;

    // int k = payload_nx;
    // int j = payload_nv;
    // plstate.segment(0, 3) = state.segment(0, 3);
    // plstate.segment(3, 3) = state.segment(3, 3);
    // plstate.segment(6, 4) = state.segment(6, 4);
    //
    // for (int i = 0; i < numOfquads; ++i) {
    //   plstate.segment(k, 3) = state.segment(k, 3);
    //   plstate.segment(k + 3 * numOfquads, 3) =
    //       state.segment(k + 3 * numOfquads, 3);
    //   k += 3;
    //   // j += 3;
    // }

    accl = Bq.inverse() * (Nq + u_inp);
    accl.segment(0, 3) -= Eigen::Vector3d(0, 0, 9.81);

    // getNextState();

    bool semi_implicit_rotation = true;
    Eigen::Vector3d xp = state.segment(0, 3);
    Eigen::Vector3d vp = state.segment(3, 3);

    if (!pointmass) {
      Eigen::Vector4d quat_p = state.segment(6, 4);
      Eigen::Vector3d wp = state.segment(10, 3);
      // [ 0.9998121  -0.007676   -0.01779678  0.00032969]
      std::cout << "accl format(OctaveFmt)" << std::endl;
      std::cout << accl.format(OctaveFmt) << std::endl;
      next_state.segment(10, 3) = accl.segment(3, 3) * dt + wp;
      if (semi_implicit_rotation) {
        wp = next_state.segment(10, 3);
      }
      next_state.segment(6, 4) = intergrate_quat_formatC(
          quat_p, wp,
          dt); // Assuming integrate_quat() is defined elsewhere in the code
    }

    next_state.segment(0, 3) = vp * dt + xp;
    next_state.segment(3, 3) = accl.segment(0, 3) * dt + vp;

    int k = payload_nx;
    int j = payload_nv;

    for (int ii = 0; ii < numOfquads; ++ii) {
      Eigen::Vector3d qi = state.segment(k, 3);
      Eigen::Vector3d wi = state.segment(k + 3 * numOfquads, 3);
      Eigen::Vector3d wdi = accl.segment(j, 3);

      next_state.segment(k + 3 * numOfquads, 3) = wdi * dt + wi;

      if (semi_implicit_rotation) {
        wi = next_state.segment(k + 3 * numOfquads, 3);
      }

      Eigen::Vector3d qdot =
          wi.cross(qi); // qdot using Eigen's cross product method
      next_state.segment(k, 3) = qdot * dt + qi;

      k += 3;
      j += 3;
    }

    for (int ii = 0; ii < numOfquads; ++ii) {
      Eigen::VectorXd tau = ctrlInputs.row(ii).segment(1, 3);
      Eigen::Vector4d curr_q = get_state_uav_i_const(ii, state).segment(0, 4);
      Eigen::Vector3d curr_w = get_state_uav_i_const(ii, state).segment(4, 3);

      Eigen::Vector4d qNext;
      Eigen::Vector3d wNext;
      uavs.at(ii).getNextAngularState(qNext, wNext, curr_w, curr_q, tau, dt);

      get_state_uav_i(ii, next_state).segment(0, 4) = qNext;
      get_state_uav_i(ii, next_state).segment(4, 3) = wNext;
    }
  }
};

Eigen::VectorXd from_stdvect_to_eigen(const std::vector<double> &vec) {
  Eigen::VectorXd out(vec.size());
  for (size_t i = 0; i < vec.size(); i++) {
    out(i) = vec.at(i);
  }
  return out;
}

// fro

void state_dynobench2coltrans(Eigen::Ref<Eigen::VectorXd> out,
                              const Eigen::Ref<const Eigen::VectorXd> in,
                              int num_uavs) {

  out = in;

  // flip quaternion of payload
  int base = 6;
  out(base) = in(base + 3);
  out(base + 1) = in(base);
  out(base + 2) = in(base + 1);
  out(base + 3) = in(base + 2);

  // from [ q, w, q, w, ...] to [ (q,q,...) , (w,...) ]
  for (size_t i = 0; i < num_uavs; i++) {
    out.segment(13 + i * 3, 3) = in.segment(13 + i * 6, 3);
    out.segment(13 + 3 * num_uavs + i * 3, 3) = in.segment(13 + i * 6 + 3, 3);
  }

  for (size_t i = 0; i < num_uavs; i++) {
    // flip the quaternion
    int base = 13 + num_uavs * 6 + 7 * i;
    out(base) = in(base + 3);
    out(base + 1) = in(base);
    out(base + 2) = in(base + 1);
    out(base + 3) = in(base + 2);
  }
}

void state_coltrans2dynobench(Eigen::Ref<Eigen::VectorXd> out,
                              const Eigen::Ref<const Eigen::VectorXd> in,
                              int num_uavs) {

  out = in;
  // flip quaternion of payload
  int base = 6;
  out(base) = in(base + 1);
  out(base + 1) = in(base + 2);
  out(base + 2) = in(base + 3);
  out(base + 3) = in(base);

  // from [ (q,q,...) , (w,...) ] to [ q, w, q, w, ...]
  for (size_t i = 0; i < num_uavs; i++) {
    out.segment(13 + i * 6, 3) = in.segment(13 + i * 3, 3);
    out.segment(13 + i * 6 + 3, 3) = in.segment(13 + 3 * num_uavs + i * 3, 3);
  }

  for (size_t i = 0; i < num_uavs; i++) {
    // flip the quaternion
    int base = 13 + num_uavs * 6 + 7 * i;
    out(base) = in(base + 1);
    out(base + 1) = in(base + 2);
    out(base + 2) = in(base + 3);
    out(base + 3) = in(base);
  }
}

void create_state_from_files(Eigen::Ref<Eigen::VectorXd> out,
                             const Eigen::Ref<const Eigen::VectorXd> payload_in,
                             std::vector<Eigen::VectorXd> uavs_states) {
  // const Eigen::Ref<const Eigen::VectorXd> cf1,
  // const Eigen::Ref<const Eigen::VectorXd> cf2) {

  int num_uavs = uavs_states.size();
  out.segment(0, 13 + num_uavs * 6) = payload_in.segment(0, 13 + num_uavs * 6);
  for (size_t i = 0; i < num_uavs; i++) {
    out.segment(13 + num_uavs * 6 + 7 * i, 7) = uavs_states.at(i).segment(6, 7);
  }
}

void state_dynobench2coltrans(Eigen::Ref<Eigen::VectorXd> out,
                              const Eigen::Ref<const Eigen::VectorXd> in) {}

BOOST_AUTO_TEST_CASE(t_big_mess) {

  using Matrix = std::vector<std::vector<double>>;
#define base_path_csv "../../test/example_2_rig_circle/"

  Matrix cfs1 = readCSV(base_path_csv "cf1.csv");
  Matrix cfs2 = readCSV(base_path_csv "cf2.csv");
  Matrix actions1 = readCSV(base_path_csv "action_1.csv");
  Matrix actions2 = readCSV(base_path_csv "action_2.csv");
  Matrix payload_state = readCSV(base_path_csv "payload.csv");

  int time_step = 300;
  // load the state
  // std::vector<UavModel> uavs(2);

  std::vector<Eigen::VectorXd> uav_states;
  std::vector<Eigen::VectorXd> uav_controls;

  uav_states.push_back(from_stdvect_to_eigen(cfs1.at(time_step)));
  uav_states.push_back(from_stdvect_to_eigen(cfs2.at(time_step)));

  uav_controls.push_back(from_stdvect_to_eigen(actions1.at(time_step)));
  uav_controls.push_back(from_stdvect_to_eigen(actions2.at(time_step)));

  std::vector<UAV> uavs(2);
  uavs.at(0).pos_fr_payload = Eigen::Vector3d(0, .3, 0);
  uavs.at(1).pos_fr_payload = Eigen::Vector3d(0, -.3, 0);

  PayloadSystem payload(uavs);

  double d = 0.046;
  double arm = 0.707106781 * d;
  double cft = 0.006;

  Eigen::Matrix4d invAll;
  invAll << 0.25, -0.25 / arm, -0.25 / arm, -0.25 / cft, 0.25, -0.25 / arm,
      0.25 / arm, 0.25 / cft, 0.25, 0.25 / arm, 0.25 / arm, -0.25 / cft, 0.25,
      0.25 / arm, -0.25 / arm, 0.25 / cft;

  Eigen::Matrix4d ctrAll = invAll.inverse();

  Eigen::MatrixXd ctrlInputs(2, 4);

  size_t i = 0;
  for (size_t i = 0; i < payload.numOfquads; i++) {
    ctrlInputs.row(i) = ctrAll * uav_controls.at(i);
  }

  payload.check();

  Eigen::VectorXd next_state(payload.nx);
  next_state.setZero();

  Eigen::VectorXd state(payload.nx);
  state.setZero();

  // state.segment(0, 13 + 2 * 6) =
  //     from_stdvect_to_eigen(payload_state.at(time_step));
  //
  // payload.get_state_uav_i(0, state).segment(0, 7) =
  //     uav_states.at(0).segment(6, 7);
  //
  // payload.get_state_uav_i(1, state).segment(0, 7) =
  //     uav_states.at(1).segment(6, 7);

  create_state_from_files(
      state, from_stdvect_to_eigen(payload_state.at(time_step)), uav_states);

  // sanity check
  {
    Eigen::VectorXd state_dyno(payload.nx);
    Eigen::VectorXd state_col_v2(payload.nx);
    state_coltrans2dynobench(state_dyno, state, payload.numOfquads);
    state_dynobench2coltrans(state_col_v2, state_dyno, payload.numOfquads);
    CHECK_LEQ((state - state_col_v2).norm(), 1e-12, "");
  }

  payload.stateEvolution(next_state, state, ctrlInputs);

  // payload.stateEvolution(ctrlInputs, uavs, uavs_params);
  Eigen::VectorXd payload_state_next =
      from_stdvect_to_eigen(payload_state.at(time_step + 1));

  CHECK_LEQ((next_state.head(payload.payload_w_cables_nx) - payload_state_next)
                .norm(),
            1e-12, AT);

  std::cout << "difference against next state Payload w cables -- this should "
               "be zero!!"
            << std::endl;
  std::cout << (next_state.head(payload.payload_w_cables_nx) -
                payload_state_next)
                   .transpose()
            << std::endl;

  std::cout
      << "pos  diff: "
      << (next_state.segment<3>(0) - payload_state_next.segment<3>(0)).norm()
      << std::endl;
  std::cout
      << "vel  diff: "
      << (next_state.segment<3>(3) - payload_state_next.segment<3>(3)).norm()
      << std::endl;

  std::cout << "quat diff: "
            << (next_state.segment<4>(6) -
                payload_state_next.segment<4>(6).normalized())
                   .norm()
            << std::endl;

  std::cout << "dif is " << std::endl;
  std::cout << next_state.segment<4>(6) - payload_state_next.segment<4>(6)
            << std::endl;

  std::cout
      << "w    diff: "
      << (next_state.segment<3>(10) - payload_state_next.segment<3>(10)).norm()
      << std::endl;

  std::cout
      << "other    diff: "
      << (next_state.segment(13, 3) - payload_state_next.segment(13, 3)).norm()
      << std::endl;

  std::cout
      << "other    diff: "
      << (next_state.segment(16, 3) - payload_state_next.segment(16, 3)).norm()
      << std::endl;

  std::cout
      << "other    diff: "
      << (next_state.segment(19, 3) - payload_state_next.segment(19, 3)).norm()
      << std::endl;

  std::cout
      << "other    diff: "
      << (next_state.segment(22, 3) - payload_state_next.segment(22, 3)).norm()
      << std::endl;

  std::cout << "lets compare uav 1" << std::endl;
  Eigen::VectorXd next1 = from_stdvect_to_eigen(cfs1.at(time_step + 1));
  Eigen::VectorXd next2 = from_stdvect_to_eigen(cfs2.at(time_step + 1));

  CHECK_LEQ((next1.segment(6, 7) -
             payload.get_state_uav_i(0, next_state).segment(0, 7))
                .norm(),
            1e-12, "");
  CHECK_LEQ((next2.segment(6, 7) -
             payload.get_state_uav_i(1, next_state).segment(0, 7))
                .norm(),
            1e-12, "");

  std::cout << "uav 1 " << std::endl;
  std::cout << "quat "
            << (next1.segment(6, 4) -
                payload.get_state_uav_i(0, next_state).segment(0, 4))
                   .norm()
            << std::endl;

  std::cout << "w "
            << (next1.segment(10, 3) -
                payload.get_state_uav_i(0, next_state).segment(4, 3))
                   .norm()
            << std::endl;

  std::cout << "uav 2 " << std::endl;
  std::cout << "quat "
            << (next2.segment(6, 4) -
                payload.get_state_uav_i(1, next_state).segment(0, 4))
                   .norm()
            << std::endl;

  std::cout << "w "
            << (next2.segment(10, 3) -
                payload.get_state_uav_i(1, next_state).segment(4, 3))
                   .norm()
            << std::endl;

  // lets compare!
  //
}
