#include <fstream>
#include <iostream>
#include <numeric>

#include "dynobench/__quadrotor_payload_rigid.hpp"

namespace dynobench {

Eigen::IOFormat OctaveFmt(Eigen::StreamPrecision, 0, ", ", ";\n", "", "", "[",
                          "]");

Eigen::Vector4d intergrate_quat_formatC(const Eigen::Vector4d &__quat_p,
                                        const Eigen::Vector3d &wp, double dt) {

  Eigen::Vector4d quat_p(__quat_p(1), __quat_p(2), __quat_p(3), __quat_p(0));

  CHECK_LEQ(std::abs(quat_p.norm() - 1.), 1e-6, "");

  Eigen::Vector4d __out, __out2;
  Eigen::Vector4d deltaQ;
  __get_quat_from_ang_vel_time(wp * dt, deltaQ, nullptr);
  quat_product(quat_p, deltaQ, __out, nullptr, nullptr);
  Eigen::Vector4d out(__out(3), __out(0), __out(1), __out(2));
  return out;
}

Eigen::Vector3d rotate_format_C(const Eigen::Vector4d &quat,
                                const Eigen::Vector3d &v) {

  // Eigen::Vector4d q = quat.normalized();

  CHECK_LEQ(std::abs(quat.norm() - 1.), 1e-6, "");
  Eigen::Matrix3d R =
      Eigen::Quaterniond(quat(0), quat(1), quat(2), quat(3)).toRotationMatrix();
  return R * v;
}

// Assuming you have a function to_matrix implemented elsewhere, it would look
// something like:
Eigen::Matrix3d to_matrix_formatC(const Eigen::Vector4d &vec) {
  // TODO: check the order of the quaternion!!

  CHECK_LEQ(std::abs(vec.norm() - 1.), 1e-6, "");

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

void UAV::getNextAngularState(Eigen::Ref<Eigen::Vector4d> next_q,
                              Eigen::Ref<Eigen::Vector3d> next_w,
                              const Eigen::Vector3d &curr_w,
                              const Eigen::Vector4d &curr_q,
                              const Eigen::Vector3d &wdot, double dt) {
  next_w = wdot * dt + curr_w;
  next_q = intergrate_quat_formatC(curr_q, curr_w, dt);
}

void UAV::getWdot(Eigen::Ref<Eigen::Vector3d> wdot,
                  const Eigen::Vector3d &curr_w, const Eigen::Vector3d &tau) {

  Eigen::Matrix3d skew_matrix = skew(curr_w);
  wdot = I.cwiseInverse().asDiagonal() *
         (tau - skew_matrix * I.asDiagonal() * curr_w);
}

void PayloadSystem::set_uavs(const std::vector<UAV> &t_uavs) {
  uavs = t_uavs;
  numOfquads = uavs.size();
  nx = 13 + numOfquads * 6 + numOfquads * 7;
  // mt = numOfquads * 0.0356 + 0.0115; // TODO: this is not general
  payload_nx = 13;
  payload_nv = 6;
  payload_w_cables_nv = payload_nv + 3 * numOfquads;
  payload_w_cables_nx = payload_nx + 6 * numOfquads;

  accl_x.resize(payload_w_cables_nv + 3 * numOfquads);
  accl_x.setZero();

  Bq.resize(payload_w_cables_nv, payload_w_cables_nv);
  Nq.resize(payload_w_cables_nv);
  u_inp.resize(payload_w_cables_nv);
}

PayloadSystem::PayloadSystem(const std::vector<UAV> &uavs) { set_uavs(uavs); }

Eigen::VectorBlock<Eigen::Ref<Eigen::VectorXd>>
PayloadSystem::get_state_uav_i(int i, Eigen::Ref<Eigen::VectorXd> state) {
  CHECK_EQ(state.size(), nx, "");
  CHECK_GEQ(i, 0, "");
  CHECK_GEQ(numOfquads, i + 1, "");
  return state.segment(payload_w_cables_nx + i * 7, 7);
}

Eigen::VectorBlock<const Eigen::Ref<const Eigen::VectorXd>>
PayloadSystem::get_state_uav_i_const(
    int i, const Eigen::Ref<const Eigen::VectorXd> &state) {
  CHECK_EQ(state.size(), nx, "");
  CHECK_GEQ(i, 0, "");
  CHECK_GEQ(numOfquads, i + 1, "");
  return state.segment(payload_w_cables_nx + i * 7, 7);
}

void PayloadSystem::getBq(Eigen::Ref<Eigen::MatrixXd> Bq,
                          const Eigen::Ref<const Eigen::VectorXd> state) {
  // Assuming payload_w_cables_nv, payload_nv, payload_nx, mt, J, state, and
  // pointmass are member variables of the class Eigen::MatrixXd
  // Bq(payload_w_cables_nv, payload_w_cables_nv);

  CHECK_EQ(state.size(), nx, "");
  CHECK_EQ(Bq.rows(), payload_w_cables_nv, "");
  CHECK_EQ(Bq.cols(), payload_w_cables_nv, "");

  Bq.setZero();

  double m_drones =
      std::accumulate(uavs.begin(), uavs.end(), 0.0,
                      [](double sum, const UAV &uav) { return sum + uav.m; });

  Bq.block<3, 3>(0, 0) = (m_payload + m_drones) * Eigen::Matrix3d::Identity();

  // std::cout << "state.transpose().format(OctaveFmt)" << std::endl;
  // std::cout << state.transpose().format(OctaveFmt) << std::endl;

  int i = payload_nv;
  int k = payload_nx;

  for (int ii = 0; ii < numOfquads; ii++) {

    // std::cout << " i "
    //           << " k " << i << " " << k << std::endl;

    double m = uavs.at(ii).m;
    double l = uavs.at(ii).l_c;

    Eigen::Vector3d qi = state.segment(k, 3);

    CHECK_LEQ(std::abs(qi.norm() - 1.), 1e-4, "");

    k += 3;

    Bq.block<3, 3>(i, 0) =
        -m * skew(qi); // Assuming skew() is defined elsewhere
    Bq.block<3, 3>(i, i) = m * l * Eigen::Matrix3d::Identity();

    if (!pointmass) {
      Eigen::Matrix3d R_p = to_matrix_formatC(
          state.segment(6, 4)); // Assuming to_matrix() is defined elsewhere
      Eigen::Vector3d posFrload(uavs.at(ii).pos_fr_payload);
      // std::cout << "posFrload.format(OctaveFmt)" << std::endl;
      // std::cout << posFrload.format(OctaveFmt) << std::endl;

      Eigen::Matrix3d qiqiT = qi * qi.transpose();

      Bq.block<3, 3>(0, 3) += -m * qiqiT * R_p * skew(posFrload);
      Bq.block<3, 3>(3, 0) += m * skew(posFrload) * R_p.transpose() * qiqiT;
      Bq.block<3, 3>(3, 3) +=
          m * skew(posFrload) * R_p.transpose() * qiqiT * R_p * skew(posFrload);
      Bq.block<3, 3>(i, 3) = m * skew(qi) * R_p * skew(posFrload);
      i += 3;
    }
  }

  if (!pointmass) {
    Bq.block<3, 3>(3, 3) = J - Bq.block<3, 3>(3, 3);
  }
}

void PayloadSystem::getNq(Eigen::Ref<Eigen::VectorXd> Nq,
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
      Nq.segment(3, 3) += skew(posFrload) * R_p.transpose() *
                          ((-m * l * wi.dot(wi) * qi) -
                           (m * qiqiT * R_p * skew(wl) * skew(wl) * posFrload));
      Nq.segment(i, 3) = m * skew(qi) * R_p * skew(wl) * skew(wl) * posFrload;
      i += 3;
    }
  }

  if (!pointmass) {
    // Assuming J is a member variable representing inertia tnsor
    Nq.segment(3, 3) -= skew(wl) * J * wl;
  }
}

void PayloadSystem::getuinp(
    Eigen::Ref<Eigen::VectorXd> u_inp,
    const Eigen::Ref<const Eigen::VectorXd> &state,
    const Eigen::Ref<const Eigen::VectorXd> &ctrlInputs) {

  CHECK_EQ(u_inp.size(), payload_w_cables_nv, "");

  int i = 0, j = payload_nv, k = payload_nx;

  for (size_t ii = 0; ii < numOfquads; ++ii) {

    // const std::string &name = paramPair.first;
    // const UAVParams &uav = paramPair.second;

    double f = ctrlInputs(ii * 4 + 0);

    // std::cout << "uavs.at(name).state.segment(6,
    // 4).transpose().format(OctaveFmt)" << std::endl; std::cout <<
    // uavs.at(name).state.segment(6, 4).transpose().format(OctaveFmt) <<
    // std::endl;

    Eigen::Vector3d u_i =
        rotate_format_C(get_state_uav_i_const(ii, state).segment<4>(0),
                        Eigen::Vector3d(0, 0, f));
    // std::cout << "u_i.format(OctaveFmt)" << std::endl;
    // std::cout << u_i.format(OctaveFmt) << std::endl;

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

void PayloadSystem::getPayloadwCablesAcceleration(Vref acc, Vcref x, Vcref u) {

  CHECK_EQ(x.size(), nx, "");

  CHECK_EQ(Bq.rows(), payload_w_cables_nv, "");
  CHECK_EQ(Bq.cols(), payload_w_cables_nv, "");

  CHECK_EQ(Nq.size(), payload_w_cables_nv, "");
  CHECK_EQ(u_inp.size(), payload_w_cables_nv, "");

  Bq.setZero();
  Nq.setZero();
  u_inp.setZero();
  getBq(Bq, x);
  getNq(Nq, x);
  getuinp(u_inp, x, u);

  // Bq.inverse() * (Nq + u_inp)
  // acc.segment(0, payload_w_cables_nv) = Bq.lu().solve(Nq + u_inp);
  acc.segment(0, payload_w_cables_nv) = Bq.inverse() * (Nq + u_inp);
  acc.segment(0, 3) -= Eigen::Vector3d(
      0, 0, 9.81); // TODO: ask Khaled -- are you sure this is correct?

  for (int ii = 0; ii < numOfquads; ++ii) {
    Eigen::Vector3d tau = u.segment(4 * ii + 1, 3);
    Eigen::Vector3d curr_w = get_state_uav_i_const(ii, x).segment(4, 3);
    Eigen::Vector3d wdot;
    uavs.at(ii).getWdot(wdot, curr_w, tau);
    acc.segment(payload_w_cables_nv + ii * 3, 3) = wdot;
  }
}

// std::pair<std::map<std::string, UavModel>, Eigen::VectorXd>
void PayloadSystem::stateEvolution(
    Eigen::Ref<Eigen::VectorXd> next_state,
    const Eigen::Ref<const Eigen::VectorXd> &state,
    const Eigen::Ref<const Eigen::VectorXd> &u, double dt) {

  CHECK_EQ(state.size(), nx, "");
  CHECK_EQ(next_state.size(), nx, "");

  getPayloadwCablesAcceleration(accl_x, state, u);

  bool semi_implicit_rotation = true;
  Eigen::Vector3d xp = state.segment(0, 3);
  Eigen::Vector3d vp = state.segment(3, 3);

  if (!pointmass) {
    Eigen::Vector4d quat_p = state.segment(6, 4);
    Eigen::Vector3d wp = state.segment(10, 3);
    next_state.segment(10, 3) = accl_x.segment(3, 3) * dt + wp;
    if (semi_implicit_rotation) {
      wp = next_state.segment(10, 3);
    }
    next_state.segment(6, 4) = intergrate_quat_formatC(
        quat_p, wp,
        dt); // Assuming integrate_quat() is defined elsewhere in the code
  }

  next_state.segment(0, 3) = vp * dt + xp;
  next_state.segment(3, 3) = accl_x.segment(0, 3) * dt + vp;

  int k = payload_nx;
  int j = payload_nv;

  for (int ii = 0; ii < numOfquads; ++ii) {
    Eigen::Vector3d qi = state.segment(k, 3);
    Eigen::Vector3d wi = state.segment(k + 3 * numOfquads, 3);
    Eigen::Vector3d wdi = accl_x.segment(j, 3);

    next_state.segment(k + 3 * numOfquads, 3) = wdi * dt + wi;

    if (semi_implicit_rotation) {
      wi = next_state.segment(k + 3 * numOfquads, 3);
    }

    Eigen::Vector3d qdot =
        wi.cross(qi); // qdot using Eigen's cross product method
    next_state.segment(k, 3) = qdot * dt + qi;
    next_state.segment(k, 3).normalize();

    k += 3;
    j += 3;
  }

  for (int ii = 0; ii < numOfquads; ++ii) {
    Eigen::Vector4d curr_q = get_state_uav_i_const(ii, state).segment(0, 4);
    Eigen::Vector3d curr_w = get_state_uav_i_const(ii, state).segment(4, 3);

    Eigen::Vector4d qNext;
    Eigen::Vector3d wNext;

    Eigen::Vector3d wdot = accl_x.segment(payload_w_cables_nv + ii * 3, 3);

    uavs.at(ii).getNextAngularState(qNext, wNext, curr_w, curr_q, wdot, dt);

    get_state_uav_i(ii, next_state).segment(0, 4) = qNext;
    get_state_uav_i(ii, next_state).segment(4, 3) = wNext;
  }

  // std::cout << "diference in coltrans " << std::endl;
  // std::cout << next_state - state << std::endl;
}

// fro

void state_dynobench2coltrans(Eigen::Ref<Eigen::VectorXd> out,
                              const Eigen::Ref<const Eigen::VectorXd> in,
                              int num_uavs) {

  out = in;

  // flip quaternion of payload
  {
    int base_dyno = 3;
    int base_coltrans = 6;
    out(base_coltrans) = in(base_dyno + 3);
    out(base_coltrans + 1) = in(base_dyno);
    out(base_coltrans + 2) = in(base_dyno + 1);
    out(base_coltrans + 3) = in(base_dyno + 2);
  }

  // copy the velocities
  { out.segment(3, 3) = in.segment(7, 3); }

  // from [ q, w, q, w, ...] to [ (q,q,...) , (w,...) ]
  for (size_t i = 0; i < num_uavs; i++) {
    out.segment(13 + i * 3, 3) = in.segment(13 + i * 6, 3);
    out.segment(13 + 3 * num_uavs + i * 3, 3) = in.segment(13 + i * 6 + 3, 3);
  }

  // [ (q,w) , (q,w) ... ] in both
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
  {
    int base_dyno = 3;
    int base_coltrans = 6;
    out(base_dyno) = in(base_coltrans + 1);
    out(base_dyno + 1) = in(base_coltrans + 2);
    out(base_dyno + 2) = in(base_coltrans + 3);
    out(base_dyno + 3) = in(base_coltrans);
  }

  { out.segment(7, 3) = in.segment(3, 3); }

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

void control_coltrans2dynobench(Eigen::Ref<Eigen::VectorXd> out,
                                const Eigen::Ref<const Eigen::VectorXd> in,
                                std::vector<UAV> uavs) {

  int num_uavs = uavs.size();

  CHECK_EQ(in.size(), 4 * num_uavs, "");
  CHECK_EQ(out.size(), 4 * num_uavs, "");

  for (size_t i = 0; i < num_uavs; i++) {
    out.segment(0 + 4 * i, 4) = uavs.at(i).invAll * in.segment(0 + 4 * i, 4) /
                                (uavs.at(i).m * 9.81 / 4.0);
  }
}

void control_dynobench2coltrans(Eigen::Ref<Eigen::VectorXd> out,
                                const Eigen::Ref<const Eigen::VectorXd> in,
                                std::vector<UAV> uavs) {

  int num_uavs = uavs.size();

  CHECK_EQ(in.size(), 4 * num_uavs, "");
  CHECK_EQ(out.size(), 4 * num_uavs, "");

  for (size_t i = 0; i < num_uavs; i++) {
    out.segment(0 + 4 * i, 4) = uavs.at(i).ctrAll * in.segment(0 + 4 * i, 4) *
                                (uavs.at(i).m * 9.81 / 4.0);
  }
}

} // namespace dynobench
