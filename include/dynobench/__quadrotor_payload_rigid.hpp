#include <fstream>
#include <iostream>

#include "math_utils.hpp"
#include <Eigen/Core>

namespace dynobench {

Eigen::Vector4d intergrate_quat_formatC(const Eigen::Vector4d &__quat_p,
                                        const Eigen::Vector3d &wp, double dt);

Eigen::Vector3d rotate_format_C(const Eigen::Vector4d &quat,
                                const Eigen::Vector3d &v);

Eigen::Matrix3d to_matrix_formatC(const Eigen::Vector4d &vec);

Eigen::Matrix3d skew(const Eigen::Vector3d &vec);

struct UAV {

  UAV() {
    invAll << 0.25, -0.25 / arm, -0.25 / arm, -0.25 / cft, 0.25, -0.25 / arm,
        0.25 / arm, 0.25 / cft, 0.25, 0.25 / arm, 0.25 / arm, -0.25 / cft, 0.25,
        0.25 / arm, -0.25 / arm, 0.25 / cft;

    ctrAll = invAll.inverse();
  }

  Eigen::Vector3d I = Eigen::Vector3d(16.571710e-6, 16.655602e-6, 29.261652e-6);
  // Eigen::Matrix3d invI = I.inverse();
  // Eigen::VectorXd states;
  double m = 0.0356;
  double l_c = .5;
  Eigen::Vector3d pos_fr_payload = Eigen::Vector3d(0., 0, 0.);

  void check() {
    // CHECK_EQ(states.size(), 7, "");
    CHECK((I.sum() > 0), "");
    CHECK((pos_fr_payload.sum() > 0), "");
    CHECK((pos_fr_payload.sum() > 1e-6), "");
  }

  void getNextAngularState(Eigen::Ref<Eigen::Vector4d> next_q,
                           Eigen::Ref<Eigen::Vector3d> next_w,
                           const Eigen::Vector3d &curr_w,
                           const Eigen::Vector4d &curr_q,
                           const Eigen::Vector3d &wdot, double dt);

  void getWdot(Eigen::Ref<Eigen::Vector3d> wdot, const Eigen::Vector3d &curr_w,
               const Eigen::Vector3d &tau);

  double d = 0.046;
  double arm = 0.707106781 * d;
  double cft = 0.006;

  Eigen::Matrix4d invAll;
  Eigen::Matrix4d ctrAll;

  void write(std::ostream &out) {

    out << "I: " << I.transpose() << std::endl;
    out << "m: " << m << std::endl;
    out << "l_c: " << l_c << std::endl;
    out << "pos_fr_payload: " << pos_fr_payload.transpose() << std::endl;
    out << "d: " << d << std::endl;
    out << "arm: " << arm << std::endl;
    out << "cft: " << cft << std::endl;
  }
};

class PayloadSystem {
public:
  PayloadSystem() = default;

  Eigen::MatrixXd Bq;
  Eigen::VectorXd Nq;
  Eigen::VectorXd u_inp;
  Eigen::VectorXd __x;

  void write(std::ostream &out) {

    out << "nx: " << nx << std::endl;
    out << "payload_nv: " << payload_nv << std::endl;
    out << "payload_nx: " << payload_nx << std::endl;
    out << "payload_w_cables_nx: " << payload_w_cables_nx << std::endl;
    out << "payload_w_cables_nv: " << payload_w_cables_nv << std::endl;
    out << "pointmass: " << pointmass << std::endl;
    out << "m_payload: " << m_payload << std::endl;
    out << "J: " << J << std::endl;
    out << "accl: " << accl << std::endl;
    out << "numOfquads: " << numOfquads << std::endl;
    out << "uavs.size(): " << uavs.size() << std::endl;

    out << "printing quad" << std::endl;
    for (size_t i = 0; i < numOfquads; i++) {
      uavs.at(i).write(out);
    }
  }

  PayloadSystem(const std::vector<UAV> &uavs);

  void ensure_state(Vref __x);

  void set_uavs(const std::vector<UAV> &uavs);

  // Eigen::VectorXd plstate;
  int payload_w_cables_nv;

  int nacc;
  int nx;
  int payload_nv;
  int payload_nx;
  int payload_w_cables_nx;
  bool pointmass = false;
  double m_payload = .0115;

  Eigen::Matrix3d J =
      Eigen::Vector3d(3.0002e-4, 7.7083e-7, 3.0075e-4).asDiagonal();

  Eigen::VectorXd accl;   // payload with cables
  Eigen::VectorXd accl_x; // full system
  int numOfquads;

  std::vector<UAV> uavs;
  // std::vector<Eigen::VectorXd> uav_states;

  // Eigen::VectorBlock<Eigen::VectorXd>
  // auto

  Eigen::VectorBlock<Eigen::Ref<Eigen::VectorXd>>
  get_state_uav_i(int i, Eigen::Ref<Eigen::VectorXd> state);

  Eigen::VectorBlock<const Eigen::Ref<const Eigen::VectorXd>>
  get_state_uav_i_const(int i, const Eigen::Ref<const Eigen::VectorXd> &state);

  void getBq(Eigen::Ref<Eigen::MatrixXd> Bq,
             const Eigen::Ref<const Eigen::VectorXd> state);

  void getPayloadwCablesAcceleration(Vref acc, Vcref x, Vcref u);

  void getNq(Eigen::Ref<Eigen::VectorXd> Nq,
             const Eigen::Ref<const Eigen::VectorXd> state);

  void getuinp(Eigen::Ref<Eigen::VectorXd> u_inp,
               const Eigen::Ref<const Eigen::VectorXd> &state,
               const Eigen::Ref<const Eigen::VectorXd> &ctrlInputs);

  void stateEvolution(Eigen::Ref<Eigen::VectorXd> next_state,
                      const Eigen::Ref<const Eigen::VectorXd> &state,
                      const Eigen::Ref<const Eigen::VectorXd> &u, double dt);

  void beautfiy_state(Eigen::Ref<Eigen::VectorXd> state) {

    std::cout << "full state" << std::endl;
    std::cout << state.transpose() << std::endl;
    std::cout << "payload pos " << std::endl;
    std::cout << state.segment<3>(0).transpose() << std::endl;
    std::cout << "payload vel " << std::endl;
    std::cout << state.segment<3>(3).transpose() << std::endl;
    std::cout << "payload quat " << std::endl;
    std::cout << state.segment<4>(6).transpose() << std::endl;
    std::cout << "payload w " << std::endl;
    std::cout << state.segment<3>(10).transpose() << std::endl;

    std::cout << "q cables" << std::endl;
    for (size_t i = 0; i < numOfquads; i++) {
      std::cout << state.segment<3>(13 + i * 3).transpose() << std::endl;
    }

    std::cout << "w cables" << std::endl;
    for (size_t i = 0; i < numOfquads; i++) {
      std::cout << state.segment<3>(13 + 3 * numOfquads + i * 3).transpose()
                << std::endl;
    }

    std::cout << "robot i " << std::endl;
    for (size_t i = 0; i < numOfquads; i++) {
      std::cout << "q " << std::endl;
      std::cout << state.segment<4>(13 + numOfquads * 6 + i * 7).transpose()
                << std::endl;
      std::cout << "w " << std::endl;
      std::cout << state.segment<3>(13 + numOfquads * 6 + i * 7 + 4).transpose()
                << std::endl;
    }
  }

  // fro
};

void state_dynobench2coltrans(Eigen::Ref<Eigen::VectorXd> out,
                              const Eigen::Ref<const Eigen::VectorXd> in,
                              int num_uavs);

void state_coltrans2dynobench(Eigen::Ref<Eigen::VectorXd> out,
                              const Eigen::Ref<const Eigen::VectorXd> in,
                              int num_uavs);

void control_coltrans2dynobench(Eigen::Ref<Eigen::VectorXd> out,
                                const Eigen::Ref<const Eigen::VectorXd> in,
                                std::vector<UAV> uavs);

void control_dynobench2coltrans(Eigen::Ref<Eigen::VectorXd> out,
                                const Eigen::Ref<const Eigen::VectorXd> in,
                                std::vector<UAV> uavs);

} // namespace dynobench
