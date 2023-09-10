#define BOOST_TEST_MODULE test_payload
#define BOOST_TEST_DYN_LINK
#include "dynobench/motions.hpp"
#include "dynobench/quadrotor_payload.hpp"
#include "dynobench/quadrotor_payload_n.hpp"
#include <boost/test/unit_test.hpp>

#define base_path "../../"

using namespace dynobench;

#include <fstream>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>

using UAVParams = std::map<std::string, double>;
Eigen::IOFormat OctaveFmt(Eigen::StreamPrecision, 0, ", ", ";\n", "", "", "[",
                          "]");

Eigen::Vector4d integrate_quat(const Eigen::Vector4d &__quat_p,
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

Eigen::Vector3d rotate(const Eigen::Vector4d &quat, const Eigen::Vector3d &v) {

  Eigen::Vector4d q = quat.normalized();
  Eigen::Matrix3d R =
      Eigen::Quaterniond(q(0), q(1), q(2), q(3)).toRotationMatrix();
  return R * v;
}

// Assuming you have a function to_matrix implemented elsewhere, it would look
// something like:
Eigen::Matrix3d to_matrix(const Eigen::Vector4d &vec) {
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

double M = 0.0356;
Eigen::Vector3d II(16.571710e-6, 16.655602e-6, 29.261652e-6);

struct UavModel {

  Eigen::VectorXd state;

  Eigen::Matrix3d I = II.asDiagonal();
  Eigen::Matrix3d invI = I.inverse();
  double dt = 0.001;

  std::pair<Eigen::Vector4d, Eigen::Vector3d>
  getNextAngularState(const Eigen::Vector3d &curr_w,
                      const Eigen::Vector4d &curr_q,
                      const Eigen::Vector3d &tau) {

    Eigen::Matrix3d skew_matrix = skew(curr_w);
    Eigen::Vector3d wdot = invI * (tau - skew_matrix * I * curr_w);
    Eigen::Vector3d wNext = wdot * dt + curr_w;
    Eigen::Vector4d qNext = integrate_quat(curr_q, curr_w, dt);

    return {qNext, wNext};
  }
};

class Payload {
public:
  Eigen::VectorXd plstate;
  int sys_dim;
  int plSysDim;
  int plStateSize;
  bool pointmass;
  double mt; // total mass of system
  Eigen::VectorXd state;
  Eigen::Matrix3d J;
  double dt;
  Eigen::VectorXd accl;
  int numOfquads;

  Eigen::MatrixXd getBq(const std::map<std::string, UAVParams> &uavs_params) {
    // Assuming sys_dim, plSysDim, plStateSize, mt, J, state, and pointmass are
    // member variables of the class
    Eigen::MatrixXd Bq(sys_dim, sys_dim);
    Bq.setZero();

    Bq.block<3, 3>(0, 0) = mt * Eigen::Matrix3d::Identity();

    int i = plSysDim;
    int k = plStateSize;

    for (const auto &[name, uav] : uavs_params) {
      double m = uav.at("m");
      double l = uav.at("l_c");

      Eigen::Vector3d qi = state.segment(k, 3);
      k += 3;

      Bq.block<3, 3>(i, 0) =
          -m * skew(qi); // Assuming skew() is defined elsewhere
      Bq.block<3, 3>(i, i) = m * l * Eigen::Matrix3d::Identity();

      if (!pointmass) {
        Eigen::Matrix3d R_p = to_matrix(
            state.segment(6, 4)); // Assuming to_matrix() is defined elsewhere
        Eigen::Vector3d posFrload(uav.at("pos_fr_payloadx"),
                                  uav.at("pos_fr_payloady"),
                                  uav.at("pos_fr_payloadz"));

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

    return Bq;
  }

  Eigen::VectorXd getNq(const std::map<std::string, UAVParams> &uavs_params) {
    // Assuming sys_dim, plSysDim, plStateSize, mt, state are member variables
    // of the class
    Eigen::VectorXd Nq(sys_dim);
    Nq.setZero();

    int i = plSysDim;
    int k = plStateSize;

    Eigen::Matrix3d Mq = mt * Eigen::Matrix3d::Identity();

    Eigen::Matrix3d R_p;
    Eigen::Vector3d wl;

    if (!pointmass) {
      R_p = to_matrix(
          state.segment(6, 4)); // Assuming to_matrix() is defined elsewhere
      wl = state.segment(10, 3);
    }

    for (const auto &[name, uav] : uavs_params) {
      double m = uav.at("m");
      double l = uav.at("l_c");

      Eigen::Vector3d qi = state.segment(k, 3);
      Eigen::Vector3d wi = state.segment(k + 3 * numOfquads, 3);

      k += 3;

      if (pointmass) {
        Nq.segment(0, 3) +=
            -m * l * wi.dot(wi) * qi; // Using Eigen's dot() for dot product
      } else {
        Eigen::Vector3d posFrload = Eigen::Vector3d(
            uav.at("pos_fr_payloadx"), uav.at("pos_fr_payloady"),
            uav.at("pos_fr_payloadz")); // Assuming pos_fr_payload is
                                        // std::vector<double>
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

    return Nq;
  }

  Eigen::VectorXd getuinp(const std::map<std::string, UAVParams> &uavs_params,
                          const Eigen::MatrixXd &ctrlInputs,
                          const std::map<std::string, UavModel> &uavs) {

    Eigen::VectorXd u_inp = Eigen::VectorXd::Zero(sys_dim);
    int i = 0, j = plSysDim, k = plStateSize;

    for (const auto &paramPair : uavs_params) {
      const std::string &name = paramPair.first;
      const UAVParams &uav = paramPair.second;

      double m = uav.at("m");
      double l = uav.at("l_c");
      double f = ctrlInputs(i, 0);

      // std::cout << "uavs.at(name).state.segment(6,
      // 4).transpose().format(OctaveFmt)" << std::endl; std::cout <<
      // uavs.at(name).state.segment(6, 4).transpose().format(OctaveFmt) <<
      // std::endl;
      Eigen::Vector3d u_i =
          rotate(uavs.at(name).state.segment(6, 4), Eigen::Vector3d(0, 0, f));
      std::cout << "u_i.format(OctaveFmt)" << std::endl;
      std::cout << u_i.format(OctaveFmt) << std::endl;

      Eigen::Vector3d posFrload;
      Eigen::Matrix3d R_p;
      Eigen::Vector3d wl;
      if (!pointmass) {
        R_p = to_matrix(state.segment(6, 4));
        wl = state.segment(10, 3);
        posFrload << uav.at("pos_fr_payloadx"), uav.at("pos_fr_payloady"),
            uav.at("pos_fr_payloadz");
      }

      Eigen::Vector3d qi = state.segment(k, 3);
      Eigen::Vector3d wi = state.segment(k + 3 * numOfquads, 3);
      k += 3;

      Eigen::Matrix3d qiqiT = qi * qi.transpose();

      u_inp.segment(0, 3) += u_i;

      Eigen::Vector3d u_perp = (Eigen::Matrix3d::Identity() - qiqiT) * u_i;
      u_inp.segment(j, 3) = -skew(qi) * u_i;

      if (!pointmass) {
        u_inp.segment(3, 3) += skew(posFrload) * R_p.transpose() * qiqiT * u_i;
      }

      i++;
      j += 3;
    }

    return u_inp;
  }

  // std::pair<std::map<std::string, UavModel>, Eigen::VectorXd>
  void stateEvolution(const Eigen::MatrixXd &ctrlInputs,
                      std::map<std::string, UavModel> &uavs,
                      const std::map<std::string, UAVParams> &uavs_params) {

    // Eigen::MatrixXd newCtrlInputs =
    //     ctrlInputs.block(1, 0, ctrlInputs.rows() - 1, ctrlInputs.cols());
    Eigen::MatrixXd Bq = getBq(uavs_params);
    Eigen::MatrixXd Nq = getNq(uavs_params);

    std::cout << "ctrlInputs.format(OctaveFmt)" << std::endl;
    std::cout << ctrlInputs.format(OctaveFmt) << std::endl;

    Eigen::VectorXd u_inp = getuinp(uavs_params, ctrlInputs, uavs);

    std::cout << "u_inp.format(OctaveFmt)" << std::endl;
    std::cout << u_inp.transpose().format(OctaveFmt) << std::endl;

    int k = plStateSize;
    int j = plSysDim;
    plstate.segment(0, 3) = state.segment(0, 3);
    plstate.segment(3, 3) = state.segment(3, 3);
    plstate.segment(6, 4) = state.segment(6, 4);

    for (int i = 0; i < numOfquads; ++i) {
      plstate.segment(k, 3) = state.segment(k, 3);
      plstate.segment(k + 3 * numOfquads, 3) =
          state.segment(k + 3 * numOfquads, 3);
      k += 3;
      j += 3;
    }

    try {
      accl = Bq.inverse() * (Nq + u_inp);
      accl.segment(0, 3) -= Eigen::Vector3d(0, 0, 9.81);
    } catch (std::exception &err) {
      std::cerr << "Unexpected error: " << err.what() << std::endl;
      ERROR_WITH_INFO("");
    }
    getNextState();

    int m = 0;
    for (auto &[id, uav] : uavs) {
      Eigen::VectorXd tau = ctrlInputs.row(m).segment(1, 3);
      Eigen::Vector4d curr_q = uav.state.segment(6, 4);
      Eigen::Vector3d curr_w = uav.state.segment(10, 3);

      Eigen::Vector4d qNext;
      Eigen::Vector3d wNext;
      std::tie(qNext, wNext) = uav.getNextAngularState(curr_w, curr_q, tau);

      uav.state.segment(6, 4) = qNext;
      uav.state.segment(10, 3) = wNext;
      m++;
    }

    // return {uavs, state};
  }

  void getNextState() {
    // Assuming state, accl, dt, sys_dim, plStateSize, plSysDim, numOfquads
    // are member variables of the class

    Eigen::Vector3d xp = state.segment(0, 3);
    Eigen::Vector3d vp = state.segment(3, 3);

    if (!pointmass) {
      Eigen::Vector4d quat_p = state.segment(6, 4);
      Eigen::Vector3d wp = state.segment(10, 3);

      std::cout << accl.format(OctaveFmt) << std::endl;
      state.segment(10, 3) = accl.segment(3, 3) * dt + wp;
      state.segment(6, 4) = integrate_quat(
          quat_p, wp,
          dt); // Assuming integrate_quat() is defined elsewhere in the code
    }

    state.segment(0, 3) = vp * dt + xp;
    state.segment(3, 3) = accl.segment(0, 3) * dt + vp;

    int k = plStateSize;
    int j = plSysDim;

    for (int i = 0; i < numOfquads; ++i) {
      Eigen::Vector3d qi = state.segment(k, 3);
      Eigen::Vector3d wi = state.segment(k + 3 * numOfquads, 3);
      Eigen::Vector3d wdi = accl.segment(j, 3);

      state.segment(k + 3 * numOfquads, 3) = wdi * dt + wi;
      Eigen::Vector3d qdot =
          wi.cross(qi); // qdot using Eigen's cross product method
      state.segment(k, 3) = qdot * dt + qi;

      k += 3;
      j += 3;
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

BOOST_AUTO_TEST_CASE(t_big_mess) {

  using Matrix = std::vector<std::vector<double>>;
  using Vector = std::vector<double>;
#define base_path_csv                                                          \
  "/home/quim/stg/khaled/col-trans/sim/example_2_rig_circle/"

  Matrix cfs1 = readCSV(base_path_csv "cf1.csv");
  Matrix cfs2 = readCSV(base_path_csv "cf2.csv");
  Matrix actions1 = readCSV(base_path_csv "action_1.csv");
  Matrix actions2 = readCSV(base_path_csv "action_2.csv");

  Matrix payload_state = readCSV(base_path_csv "payload.csv");

  int time_step = 300;
  // load the state
  // std::vector<UavModel> uavs(2);

  std::map<std::string, UavModel> uavs;

  uavs.insert({"uav_1", UavModel()});
  uavs.insert({"uav_2", UavModel()});

  std::vector<Eigen::VectorXd> uav_states;
  std::vector<Eigen::VectorXd> uav_controls;

  uav_states.push_back(from_stdvect_to_eigen(cfs1.at(time_step)));
  uav_states.push_back(from_stdvect_to_eigen(cfs2.at(time_step)));

  uav_controls.push_back(from_stdvect_to_eigen(actions1.at(time_step)));
  uav_controls.push_back(from_stdvect_to_eigen(actions2.at(time_step)));

  Payload payload;
  payload.state = from_stdvect_to_eigen(payload_state.at(time_step));

  std::map<std::string, std::map<std::string, double>> uavs_params;

  UAVParams uav_params_1{
      {"m", M},
      {"l_c", .5},
      {"pos_fr_payloadx", 0.},
      {"pos_fr_payloady", 0.3},
      {"pos_fr_payloadz", 0.},
  };

  UAVParams uav_params_2{
      {"m", M},
      {"l_c", .5},
      {"pos_fr_payloadx", 0.},
      {"pos_fr_payloady", -0.3},
      {"pos_fr_payloadz", 0.},

  };

  uavs_params.insert({"uav_1", uav_params_1});
  uavs_params.insert({"uav_2", uav_params_2});

  double d = 0.046;
  double arm = 0.707106781 * d;
  double cft = 0.006;

  Eigen::Matrix4d invAll;
  invAll << 0.25, -0.25 / arm, -0.25 / arm, -0.25 / cft, 0.25, -0.25 / arm,
      0.25 / arm, 0.25 / cft, 0.25, 0.25 / arm, 0.25 / arm, -0.25 / cft, 0.25,
      0.25 / arm, -0.25 / arm, 0.25 / cft;

  Eigen::Matrix4d ctrAll = invAll.inverse();

  Eigen::MatrixXd ctrlInputs(2, 4);

  payload.state = from_stdvect_to_eigen(payload_state.at(time_step));
  payload.state.segment(6, 4).normalize();
  payload.plstate = payload.state; // what the hell is plstate?
  payload.mt = 2. * M + 0.0115;
  payload.plStateSize = 13;
  payload.dt = 0.001;

  payload.plSysDim = 6;
  payload.sys_dim = payload.plSysDim + 3 * 2;
  payload.numOfquads = 2;
  payload.accl = Eigen::VectorXd::Zero(6 + 3 * 2);
  payload.J = Eigen::Vector3d(3.0002e-4, 7.7083e-7, 3.0075e-4).asDiagonal();

  // payload.state_size = payload.plStateSize + 6*2;

  // self.state_size = self.plStateSize + 6*self.numOfquads
  // #13 for the payload and (3+3)*n for each cable angle and its derivative

  size_t i = 0;
  for (auto &[k, v] : uavs) {
    v.state = uav_states.at(i);
    ctrlInputs.row(i) = ctrAll * uav_controls.at(i);
    i++;
  }

  payload.stateEvolution(ctrlInputs, uavs, uavs_params);

  Eigen::VectorXd loadState = payload.state;
  Eigen::VectorXd payload_state_next =
      from_stdvect_to_eigen(payload_state.at(time_step + 1));

  std::cout << "difference against next state -- this should be zero!!"
            << std::endl;
  std::cout << (payload_state_next - loadState).transpose() << std::endl;

  std::cout
      << "pos  diff: "
      << (loadState.segment<3>(0) - payload_state_next.segment<3>(0)).norm()
      << std::endl;
  std::cout
      << "vel  diff: "
      << (loadState.segment<3>(3) - payload_state_next.segment<3>(3)).norm()
      << std::endl;

  std::cout << "quat diff: "
            << (loadState.segment<4>(6) -
                payload_state_next.segment<4>(6).normalized())
                   .norm()
            << std::endl;

  std::cout << "dif is " << std::endl;
  std::cout << loadState.segment<4>(6) - payload_state_next.segment<4>(6) << std::endl;

  std::cout
      << "w    diff: "
      << (loadState.segment<3>(10) - payload_state_next.segment<3>(10)).norm()
      << std::endl;


  std::cout
      << "other    diff: "
      << (loadState.segment(13,3) - payload_state_next.segment(13,3)).norm()
      << std::endl;

  std::cout
      << "other    diff: "
      << (loadState.segment(16,3) - payload_state_next.segment(16,3)).norm()
      << std::endl;

  std::cout
      << "other    diff: "
      << (loadState.segment(19,3) - payload_state_next.segment(19,3)).norm()
      << std::endl;

  std::cout
      << "other    diff: "
      << (loadState.segment(22,3) - payload_state_next.segment(22,3)).norm()
      << std::endl;




  std::cout << "lets compare uav 1" << std::endl;

  Eigen::VectorXd next1 = from_stdvect_to_eigen(cfs1.at(time_step + 1));
  Eigen::VectorXd next2 = from_stdvect_to_eigen(cfs2.at(time_step + 1));

  std::cout << "uav 1 " << std::endl;
  std::cout
      << "quat "
      << (next1.segment(6, 4) - uavs.at("uav_1").state.segment(6, 4)).norm()
      << std::endl;

  std::cout
      << "w "
      << (next1.segment(10, 3) - uavs.at("uav_1").state.segment(10, 3)).norm()
      << std::endl;

  std::cout << "uav 2 " << std::endl;
  std::cout
      << "quat "
      << (next2.segment(6, 4) - uavs.at("uav_2").state.segment(6, 4)).norm()
      << std::endl;

  std::cout
      << "w "
      << (next2.segment(10, 3) - uavs.at("uav_2").state.segment(10, 3)).norm()
      << std::endl;

  // lets compare!
}
