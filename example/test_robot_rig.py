import sys
sys.path.append('./')
# sys.path.append('../')
import robot_python
import numpy as np
import yaml
from pathlib import Path
np.set_printoptions(linewidth=np.inf)
np.set_printoptions(suppress=True)

DYNOBENCH_MODELS_PATH = "/home/khaledwahba94/imrc/payload-uavs-planner/coltrans-planning/deps/dynoplan/dynobench/models/"

DATA_PATH = "/home/khaledwahba94/imrc/payload-uavs-planner/coltrans-planning/deps/dynoplan/dynobench/test/example_2_rig_circle"

class PayloadRobot:
    def __init__(self, model_path):
        self.model_path = DYNOBENCH_MODELS_PATH + model_path 
        self.__loadyaml()
        self.robot = robot_python.robot_factory(DYNOBENCH_MODELS_PATH + "{}_{}.yaml".format(self.payloadType,self.num_robots), [], [])
     
    def __loadyaml(self):
        if self.model_path is not None:
            with open(self.model_path, "r") as f:
                model_params = yaml.safe_load(f)
                self.num_robots = model_params["num_robots"]
                if model_params["point_mass"]:
                    self.payloadType = "point"
                else:
                    self.payloadType = "rigid"
            print("model loaded successfuly\n")
        else: 
            raise ValueError("Model path is none")
    

def correctFullStateOrder(pl, uavs, num_robots):
    # flip quat of payload
    quat = pl[6:10]
    quat_flipped = np.array([quat[1], quat[2], quat[3], quat[0]])

    plstate_ordered = np.concatenate([pl[0:3], quat_flipped, pl[3:6], pl[10:13]])
    # reorder cable states
    c_state = np.zeros(6*num_robots)
    for i in range(num_robots):
        c_state[6*i: 6*i + 3] = pl[13 + 3*i : 13+ 3*i + 3]
        c_state[6*i + 3: 6*i + 6] = pl[13 + 3*num_robots + 3*i: 13 + 3*num_robots + 3*i + 3]
    quat_uavs = []
    omegas = []
    uav_angstate = np.zeros(7*num_robots)
    for uav,i in zip(uavs, range(num_robots)): 
        quat_uavs_tmp = uav[6:10]
        quat_uavs_flipped = np.array([quat_uavs_tmp[1], quat_uavs_tmp[2], quat_uavs_tmp[3], quat_uavs_tmp[0]])
        om_tmp = uav[10:13]
        uav_angstate[7*i: 7*i + 7] = np.concatenate([quat_uavs_flipped, om_tmp])
    fullstate = np.concatenate([plstate_ordered, c_state, uav_angstate])
    return fullstate

def main():
    robotClass = PayloadRobot("rigid_2.yaml")
    robot = robotClass.robot

    ## load data
    # uavs states 
    with open(DATA_PATH +"/cf1.csv") as f:
        cf1 = np.loadtxt(f, delimiter=',')
    with open(DATA_PATH +"/cf2.csv") as f:
        cf2 = np.loadtxt(f, delimiter=",")

    # load actions: motor forces divided by u_nominal: m*9.81
    with open(DATA_PATH +"/action_1.csv") as f:
        u1 = np.loadtxt(f, delimiter=',')
    with open(DATA_PATH +"/action_2.csv") as f:
        u2 = np.loadtxt(f, delimiter=",")
    # load the payload states: 
    # payload st: position velocity, quat [qw, qx, qy, qz], ang vel
    # cable st: q1, q2, w1, w2 
    # payload full state in csv: pos, vel, qaut, w, q1, q2, w1, w2
    
    with open(DATA_PATH +"/payload.csv") as f:
        payload_state = np.loadtxt(f, delimiter=',')

    i = 300
    j = i+1
    p_i = payload_state[i,:]
    cf_i = np.array([cf1[i], cf2[i]])

    action_i = np.concatenate([np.array(u1[i]), np.array(u2[i])])

    p_j = payload_state[j,:]
    cf_j = np.array([cf1[j], cf2[j]])
    action_j = np.concatenate([u1[j], u2[j]])
    
    # print("uavs_states", cf_0)
    x_i = correctFullStateOrder(p_i, cf_i, robotClass.num_robots)
    xnext = np.zeros_like(x_i)
    robot.step(xnext, x_i, action_i, 1e-3)
    x_j = correctFullStateOrder(p_j, cf_j, robotClass.num_robots)
    print(xnext - x_j)
    print()
    print(xnext)
    print()
    print(x_j)
if __name__ == "__main__":
    main()