import mujoco
import mujoco.viewer
import numpy as np

# LOAD MODEL

model = mujoco.MjModel.from_xml_path("/Users/cristianvoltan/Desktop/unipd/tirocinio/CAD/electrolux/electrolux.xml")
data = mujoco.MjData(model)

# ACTUATORS

arm_actuators    = [0, 1, 2, 3, 4, 5, 6]
gripper_actuator = 7

# JOINT-SPACE IMPEDANCE CONTROL
# The arm actuators are configured as joint-space impedance controllers:
#
# tau = Kq * (q_des - q) - Dq * qdot
#
# where:
# q_des is sent through data.ctrl[arm_actuators]
# Kq is the joint stiffness
# Dq is the joint damping
#
# In MuJoCo:
# gainprm[0] = Kq
# biasprm[1] = -Kq
# biasprm[2] = -Dqok 

K_APPROACH = 1200.0   
D_APPROACH = 180.0

#K_GRASP = 2000.0
#D_GRASP = 300.0

K_GRASP = 800.0
D_GRASP = 120.0

#K_OPENING = 2500.0
#D_OPENING = 350.0

K_OPENING = 400.0 
D_OPENING = 60.0

#K_CLOSING = 2000.0
#D_CLOSING = 300.0

K_CLOSING = 600.0
D_CLOSING = 90.0

#K_HOLD = 2500.0
#D_HOLD = 350.0

K_HOLD = 800.0
D_HOLD = 120.0

JOINT_FORCE_LIMITS = np.array([87, 87, 87, 87, 12, 12, 12])
# JOINT_FORCE_LIMITS = np.array([1500, 1500, 1500, 1500, 1500, 1500, 1500])


def set_joint_impedance(Kq, Dq, force_limit):
    """
    Set joint-space impedance parameters.

    Equivalent behavior:

        tau = Kq * (q_des - q) - Dq * qdot

    q_des is given by data.ctrl.
    """

    for i,a in enumerate(arm_actuators):
        model.actuator_gainprm[a, 0] = Kq
        model.actuator_biasprm[a, 1] = -Kq
        model.actuator_biasprm[a, 2] = -Dq

        model.actuator_forcerange[a, 0] = -force_limit[i]
        model.actuator_forcerange[a, 1] =  force_limit[i]


# Initial impedance setting
set_joint_impedance(K_APPROACH, D_APPROACH, JOINT_FORCE_LIMITS)


# JOINTS

all_joint_names = ["joint1", "joint2", "joint3", "joint4", "joint5", "joint6", "joint7"]

ik_joint_names = ["joint1", "joint2", "joint3", "joint4", "joint5"]


all_qposadr = [] # joint position
all_dofadr  = [] # joint velocity

for name in all_joint_names:
    jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)
    all_qposadr.append(model.jnt_qposadr[jid])
    all_dofadr.append(model.jnt_dofadr[jid])

all_qposadr = np.array(all_qposadr)
all_dofadr  = np.array(all_dofadr)


ik_qposadr = []
ik_dofadr  = []

for name in ik_joint_names:
    jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)
    ik_qposadr.append(model.jnt_qposadr[jid])
    ik_dofadr.append(model.jnt_dofadr[jid])

ik_qposadr = np.array(ik_qposadr)
ik_dofadr  = np.array(ik_dofadr)


# DOOR JOINT
door_joint_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "giunto_porta")
door_qposadr = model.jnt_qposadr[door_joint_id] # door position
door_dofadr = model.jnt_dofadr[door_joint_id]   # door velocity

model.dof_frictionloss[door_dofadr] = 65.0

# SITES AND WELD
site_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, "grasp_site")
target_site_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, "target_maniglia")
weld_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_EQUALITY, "presa")


# FIXED WRIST JOINTS
joint6_index = all_joint_names.index("joint6")
joint7_index = all_joint_names.index("joint7")

joint6_angle = 2.56
joint7_angle = 0.811


# HOME POSITION

q_home = np.array([0.811, 0.3, 0.0, -1.02, 0.0, joint6_angle, joint7_angle])

for i in range(len(all_joint_names)):
    data.qpos[all_qposadr[i]] = q_home[i]   # set initial joint posistions

data.ctrl[arm_actuators] = q_home
data.ctrl[gripper_actuator] = 255

mujoco.mj_forward(model, data) # update model with home position


# Desired joint configurations
q_des_all = data.qpos[all_qposadr].copy() 
q_des_ik  = data.qpos[ik_qposadr].copy()


# TRAJECTORY

trajectory = np.load("/Users/cristianvoltan/Desktop/unipd/tirocinio/CAD/electrolux/traiettorie/traiettoria_apertura.npy")

OPEN_DEG = 62

n_points_open = int(len(trajectory) * OPEN_DEG / 180)
n_points_open = max(2, min(n_points_open, len(trajectory)))

trajectory_close = trajectory[:n_points_open][::-1].copy()

target_pos = trajectory[0, 0:3] # initial position of trajectory


# STATE MACHINE

APPROACH_HANDLE = 0
CLOSE_GRIPPER   = 1
STABILIZE_WELD  = 2
OPEN_DOOR       = 3
WAIT_OPEN       = 4
CLOSE_DOOR      = 5
HOLD_CLOSED     = 6

# NUM_CYCLES
NUM_CYCLES = 2
current_cycle = 0

# VARIABLES

q_des_all = data.qpos[all_qposadr].copy()
q_des_all[joint6_index] = joint6_angle
q_des_all[joint7_index] = joint7_angle

q_des_ik = data.qpos[ik_qposadr].copy()

# gain for IK updates
gain_phase0 = 0.01
gain_phase3 = 0.06
gain_close  = 0.06

damping  = 5e-3
ik_every = 1

phase       = APPROACH_HANDLE
phase_timer = 0

CLOSE_DURATION = 500

traj_index      = 1
close_index     = 0
TRAJ_WAIT       = 4  # update IK target every TRAJ_WAIT steps
traj_wait_timer = 0

best_dist     = float("inf") 
stall_counter = 0
STALL_LIMIT   = 2000

WAIT_BEFORE_CLOSE_SEC   = 4.0
WAIT_BEFORE_CLOSE_STEPS = int(WAIT_BEFORE_CLOSE_SEC / model.opt.timestep)


# PI PARAMETERS FOR CARTESIAN IK
# This PI is only used to generate q_des through inverse kinematics.

KP = 1.20
KI = 0.10
INTEGRAL_CLIP = 0.05

error_integral = np.zeros(3)


def reset_integral():
    """
    Reset Cartesian integral error.
    """

    global error_integral
    error_integral[:] = 0.0


# INVERSE KINEMATICS WITH PI
def solve_position_ik(target, gain):
    """
    Compute desired joint positions for the first five joints.

    The IK generates q_des_ik.
    Then q_des_ik is used as equilibrium position for the
    joint-space impedance controller.
    """

    global q_des_ik, error_integral

    mujoco.mj_forward(model, data)

    site_pos  = data.site_xpos[site_id].copy()
    error_pos = target - site_pos

    # Integral update with anti-windup
    error_integral += error_pos * model.opt.timestep
    error_integral  = np.clip(error_integral, -INTEGRAL_CLIP,INTEGRAL_CLIP)

    control = KP * error_pos + KI * error_integral

    jacp = np.zeros((3, model.nv))
    mujoco.mj_jacSite(model, data, jacp, None, site_id)

    J = jacp[:, ik_dofadr]

    A  = J @ J.T + damping * np.eye(3)
    dq = J.T @ np.linalg.solve(A, control)

    q_des_ik += gain * dq

    # Joint limits
    for i, name in enumerate(ik_joint_names):

        jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)
        if model.jnt_limited[jid]:
            q_min, q_max = model.jnt_range[jid]
            q_des_ik[i] = np.clip(q_des_ik[i], q_min, q_max)

    return np.linalg.norm(error_pos)


# SIMULATION
MAX_STEPS = 30000 * NUM_CYCLES

with mujoco.viewer.launch_passive(model, data) as viewer:

    for step in range(MAX_STEPS):

        match phase:

            # PHASE 0: APPROACH HANDLE
            case 0:

                set_joint_impedance(K_APPROACH,D_APPROACH,JOINT_FORCE_LIMITS)

                if step % ik_every == 0:

                    dist = solve_position_ik(target_pos,gain_phase0)
                    q_des_all[0:5] = q_des_ik
                    q_des_all[5] = joint6_angle
                    q_des_all[6] = joint7_angle

                    if dist < best_dist - 0.001:
                        best_dist = dist
                        stall_counter = 0
                    else:
                        stall_counter += 1

                    if dist < 0.005 or stall_counter >= STALL_LIMIT:
                        reset_integral()
                        phase = CLOSE_GRIPPER
                        phase_timer = 0


            # PHASE 1: CLOSE GRIPPER
            case 1:

                set_joint_impedance(K_GRASP,D_GRASP,JOINT_FORCE_LIMITS)

                phase_timer += 1

                if phase_timer >= CLOSE_DURATION:

                    # reset velocities and accelerations
                    data.qvel[:] = 0.0
                    data.qacc[:] = 0.0

                    mujoco.mj_forward(model, data)

                    data.eq_active[weld_id] = 1

                    mujoco.mj_forward(model, data)

                    data.qvel[:] = 0.0
                    data.qacc[:] = 0.0
                    mujoco.mj_forward(model, data)

                    reset_integral()

                    phase = STABILIZE_WELD
                    phase_timer = 0


            # PHASE 2: STABILIZE WELD
            case 2:

                set_joint_impedance(K_GRASP,D_GRASP, JOINT_FORCE_LIMITS)

                phase_timer += 1

                data.eq_active[weld_id] = 1

                if phase_timer >= 25000:
                    data.qvel[:] = 0.0
                    data.qacc[:] = 0.0
                    mujoco.mj_forward(model, data)

                    traj_index = 1
                    traj_wait_timer = 0

                    reset_integral()

                    phase = OPEN_DOOR


            # PHASE 3: OPEN DOOR
            case 3:

                set_joint_impedance(K_OPENING, D_OPENING, JOINT_FORCE_LIMITS)

                if step % ik_every == 0:

                    data.eq_active[weld_id] = 1

                    door_angle_deg_now = np.rad2deg(data.qpos[door_qposadr])
                    if door_angle_deg_now < 3.0:
                        model.dof_frictionloss[door_dofadr] = 62.06
                    else:
                        model.dof_frictionloss[door_dofadr] = 2.0

                    target_pos_traj = trajectory[traj_index, 0:3]

                    dist = solve_position_ik(target_pos_traj,gain_phase3)

                    q_des_all[0:5] = q_des_ik
                    q_des_all[5] = joint6_angle
                    q_des_all[6] = joint7_angle

                    if dist < 0.03:

                        traj_wait_timer += 1

                        if (traj_wait_timer >= TRAJ_WAIT and traj_index < n_points_open - 1):
                            traj_index += 1
                            traj_wait_timer = 0
                            reset_integral()

                    else:
                        traj_wait_timer = 0

                    if traj_index >= n_points_open - 1 and dist < 0.03:
                        door_angle = np.rad2deg(data.qpos[door_qposadr])

                        if door_angle >= 60.0: 

                            reset_integral()
                            phase = WAIT_OPEN
                            phase_timer = 0


            # PHASE 4: WAIT WITH DOOR OPEN
            case 4:

                set_joint_impedance(K_HOLD, D_HOLD, JOINT_FORCE_LIMITS)

                data.eq_active[weld_id] = 1

                phase_timer += 1

                if phase_timer >= WAIT_BEFORE_CLOSE_STEPS:

                    data.qvel[:] = 0.0
                    data.qacc[:] = 0.0

                    mujoco.mj_forward(model, data)

                    close_index = 0
                    traj_wait_timer = 0
                    reset_integral()

                    phase = CLOSE_DOOR

            # PHASE 5: CLOSE DOOR
            case 5:

                set_joint_impedance(K_CLOSING, D_CLOSING, JOINT_FORCE_LIMITS)

                if step % ik_every == 0:

                    data.eq_active[weld_id] = 1

                    target_pos_close = trajectory_close[close_index, 0:3]

                    dist = solve_position_ik(target_pos_close,gain_close)

                    q_des_all[0:5] = q_des_ik
                    q_des_all[5] = joint6_angle
                    q_des_all[6] = joint7_angle

                    if dist < 0.03:

                        traj_wait_timer += 1

                        if (traj_wait_timer >= TRAJ_WAIT and close_index < len(trajectory_close) - 1):
                            close_index += 1
                            traj_wait_timer = 0
                            reset_integral()

                    else:
                        traj_wait_timer = 0

                    if close_index >= len(trajectory_close) - 1 and dist < 0.03:

                        reset_integral()

                        phase = HOLD_CLOSED
                        phase_timer = 0

            # PHASE 6: HOLD CLOSED
            case 6:

                set_joint_impedance(K_HOLD,D_HOLD,JOINT_FORCE_LIMITS)

                data.eq_active[weld_id] = 1

                phase_timer += 1

                HOLD_STEPS = int(15.0 / model.opt.timestep)

                if phase_timer >= HOLD_STEPS:
                    current_cycle += 1
                    print(f"=== Ciclo {current_cycle}/{NUM_CYCLES} completato ===")

                    if current_cycle >= NUM_CYCLES:
                        break

                    # reset for next cycle
                    traj_index = 1
                    close_index = 0
                    traj_wait_timer = 0
                    data.qvel[:] = 0.0  
                    data.qacc[:] = 0.0
                    mujoco.mj_forward(model, data)
                    reset_integral()

                    phase = OPEN_DOOR
                    phase_timer = 0
                    model.dof_frictionloss[door_dofadr] = 62.06


        # SEND DESIRED JOINT CONFIGURATION
        q_des_all[5] = joint6_angle
        q_des_all[6] = joint7_angle

        data.ctrl[arm_actuators] = q_des_all

        # GRIPPER CONTROL
        if phase == APPROACH_HANDLE:
            data.ctrl[gripper_actuator] = 255
        else:
            data.ctrl[gripper_actuator] = 50


        # STEP SIMULATION
        mujoco.mj_step(model, data)
        viewer.sync()

        # PRINT INFORMATION
        if step % 50 == 0:
            door_angle_deg = np.rad2deg(data.qpos[door_qposadr])
            time_sec = step * model.opt.timestep
            joint_torques = data.qfrc_actuator[all_dofadr]
            REAL_LIMITS = np.array([87, 87, 87, 87, 12, 12, 12])
            violations = np.abs(joint_torques) > REAL_LIMITS

            print(f"Tempo {time_sec:.2f} | " f"fase {phase} | " f"apertura porta: {door_angle_deg:.2f}° | " f"tau: {np.round(joint_torques, 2)} | " f"{'*** VIOLAZIONE J' + str(np.where(violations)[0] + 1) + ' ***' if violations.any() else 'ok'}")


    door_angle_deg = np.rad2deg(data.qpos[door_qposadr])
    print(f"Apertura porta finale: {door_angle_deg:.2f}°")