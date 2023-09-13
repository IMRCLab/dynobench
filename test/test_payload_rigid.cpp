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

#include "dynobench/__quadrotor_payload_rigid.hpp"

#define base_path "../../"

using namespace dynobench;

using UAVParams = std::map<std::string, double>;

Eigen::VectorXd from_stdvect_to_eigen(const std::vector<double> &vec) {
  Eigen::VectorXd out(vec.size());
  for (size_t i = 0; i < vec.size(); i++) {
    out(i) = vec.at(i);
  }
  return out;
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

  std::vector<int> time_steps = {300, 310};

  std::vector<UAV> uavs(2);
  uavs.at(0).pos_fr_payload = Eigen::Vector3d(0, .3, 0);
  uavs.at(1).pos_fr_payload = Eigen::Vector3d(0, -.3, 0);

  PayloadSystem payload(uavs);

  for (size_t t = 0; t < time_steps.size(); t++) {
    time_step = time_steps.at(t);

    // load the state
    // std::vector<UavModel> uavs(2);

    std::vector<Eigen::VectorXd> uav_states;
    std::vector<Eigen::VectorXd> uav_controls;

    uav_states.push_back(from_stdvect_to_eigen(cfs1.at(time_step)));
    uav_states.push_back(from_stdvect_to_eigen(cfs2.at(time_step)));

    uav_controls.push_back(from_stdvect_to_eigen(actions1.at(time_step)));
    uav_controls.push_back(from_stdvect_to_eigen(actions2.at(time_step)));

    Eigen::VectorXd u(8);

    for (size_t i = 0; i < payload.numOfquads; i++) {
      u.segment(i * 4, 4) = payload.uavs.at(i).ctrAll * uav_controls.at(i);
    }

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
    double dt = 1e-3;

    payload.stateEvolution(next_state, state, u, dt);

    // payload.stateEvolution(ctrlInputs, uavs, uavs_params);
    Eigen::VectorXd payload_state_next =
        from_stdvect_to_eigen(payload_state.at(time_step + 1));

    std::cout
        << "difference against next state Payload w cables -- this should "
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

    std::cout << "w    diff: "
              << (next_state.segment<3>(10) - payload_state_next.segment<3>(10))
                     .norm()
              << std::endl;

    std::cout << "other    diff: "
              << (next_state.segment(13, 3) - payload_state_next.segment(13, 3))
                     .norm()
              << std::endl;

    std::cout << "other    diff: "
              << (next_state.segment(16, 3) - payload_state_next.segment(16, 3))
                     .norm()
              << std::endl;

    std::cout << "other    diff: "
              << (next_state.segment(19, 3) - payload_state_next.segment(19, 3))
                     .norm()
              << std::endl;

    std::cout << "other    diff: "
              << (next_state.segment(22, 3) - payload_state_next.segment(22, 3))
                     .norm()
              << std::endl;

    std::cout << "lets compare uav 1" << std::endl;
    Eigen::VectorXd next1 = from_stdvect_to_eigen(cfs1.at(time_step + 1));
    Eigen::VectorXd next2 = from_stdvect_to_eigen(cfs2.at(time_step + 1));

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

    CHECK_LEQ(
        (next_state.head(payload.payload_w_cables_nx) - payload_state_next)
            .norm(),
        1e-12, AT);

    CHECK_LEQ((next1.segment(6, 7) -
               payload.get_state_uav_i(0, next_state).segment(0, 7))
                  .norm(),
              1e-12, "");
    CHECK_LEQ((next2.segment(6, 7) -
               payload.get_state_uav_i(1, next_state).segment(0, 7))
                  .norm(),
              1e-12, "");
  }

  // lets compare!
  //
}
