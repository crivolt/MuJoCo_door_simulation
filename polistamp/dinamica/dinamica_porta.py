import mujoco
import mujoco.viewer
import numpy as np

# load model and create data
model = mujoco.MjModel.from_xml_path("/Users/cristianvoltan/Desktop/unipd/tirocinio/CAD/polistamp/polistamp.xml")
data = mujoco.MjData(model)

arm_actuators    = [0, 1, 2, 3, 4, 5, 6]
gripper_actuator = 7

# arms actuator

ARM_KP = 2500.0
ARM_KV = 350.0
ARM_FORCE = 1500.0

for a in arm_actuators:
    model.actuator_gainprm[a, 0] = ARM_KP
    model.actuator_biasprm[a, 1] = -ARM_KP
    model.actuator_biasprm[a, 2] = -ARM_KV

    model.actuator_forcerange[a, 0] = -ARM_FORCE
    model.actuator_forcerange[a, 1] = ARM_FORCE


all_joint_names = ["joint1", "joint2", "joint3", "joint4", "joint5", "joint6", "joint7"]
ik_joint_names  = ["joint1", "joint2", "joint3", "joint4", "joint5"]

all_qposadr, all_dofadr = [], []

for name in all_joint_names:
    jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)
    all_qposadr.append(model.jnt_qposadr[jid])
    all_dofadr.append(model.jnt_dofadr[jid])

all_qposadr = np.array(all_qposadr)
all_dofadr  = np.array(all_dofadr)

ik_qposadr, ik_dofadr = [], []

for name in ik_joint_names:
    jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)
    ik_qposadr.append(model.jnt_qposadr[jid])
    ik_dofadr.append(model.jnt_dofadr[jid])

ik_qposadr = np.array(ik_qposadr)
ik_dofadr  = np.array(ik_dofadr)

# door joint

door_joint_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "giunto_porta")
door_qposadr = model.jnt_qposadr[door_joint_id]

handle_joint_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "giunto_maniglia")
handle_qposadr = model.jnt_qposadr[handle_joint_id]

# sites

site_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, "grasp_site")
target_site_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, "target_maniglia")
weld_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_EQUALITY, "presa")


# fixed joint

joint6_index = all_joint_names.index("joint6")
joint7_index = all_joint_names.index("joint7")

joint6_angle = 2.43
joint7_angle = 0.811

# HOME POSITION — porta il braccio vicino alla maniglia prima che parta l'IK
q_home = np.array([0.869, 0.688, 0.0, -0.4, 0.0, joint6_angle, joint7_angle])

for i in range(len(all_joint_names)):
    data.qpos[all_qposadr[i]] = q_home[i]

data.ctrl[arm_actuators] = q_home
data.ctrl[gripper_actuator] = 255
mujoco.mj_forward(model, data)

# aggiorna q_des con la home, non con zero
q_des_all = data.qpos[all_qposadr].copy()
q_des_ik  = data.qpos[ik_qposadr].copy()

# trajectory

trajectory = np.load("/Users/cristianvoltan/Desktop/unipd/tirocinio/CAD/polistamp/traiettorie/traiettoria_apertura.npy")
trajectory_handle = np.load("/Users/cristianvoltan/Desktop/unipd/tirocinio/CAD/polistamp/traiettorie/traiettoria_apertura_maniglia.npy")

OPEN_DEG = 62

n_points_open = int(len(trajectory) * OPEN_DEG / 180)
n_points_open = max(2, min(n_points_open, len(trajectory)))

trajectory_close = trajectory[:n_points_open][::-1].copy() # closing trajectory

target_pos = trajectory_handle[0, 0:3]

# state machine

APPROACH_HANDLE  = 0
CLOSE_GRIPPER    = 1
STABILIZE_WELD   = 2
OPEN_HANDLE      = 3
OPEN_DOOR        = 4
WAIT_OPEN        = 5
CLOSE_DOOR       = 6
HOLD_CLOSED      = 7

# variables

q_des_all = data.qpos[all_qposadr].copy()
q_des_all[joint6_index] = joint6_angle
q_des_all[joint7_index] = joint7_angle

q_des_ik = data.qpos[ik_qposadr].copy()

gain_phase0 = 0.002
gainn_handle = 0.5
gain_phase3 = 0.050
gain_close  = 0.050

damping  = 5e-3 # for avoid singularities in IK
ik_every = 1 # compute IK every N steps

phase       = APPROACH_HANDLE
phase_timer = 0

CLOSE_DURATION = 300

handle_index = 1
traj_index      = 1 # start from the second point, first is initial position
close_index     = 0
TRAJ_WAIT       = 4 # steps before moving to next point
traj_wait_timer = 0

best_dist     = float("inf") # best distance at beginnig
stall_counter = 0 # fail to improve distance
STALL_LIMIT   = 2000

WAIT_BEFORE_CLOSE_SEC   = 13.0
WAIT_BEFORE_CLOSE_STEPS = int(WAIT_BEFORE_CLOSE_SEC / model.opt.timestep)


# PI parameters

KP             = 1.20    
KI             = 0.10  
INTEGRAL_CLIP  = 0.05   # anti-windup

error_integral = np.zeros(3)


def reset_integral():
    """Reset integral error to zero"""
    global error_integral
    error_integral[:] = 0.0


# Inverse Kinematics with PI

def solve_position_ik(target, gain):
    global q_des_ik, error_integral # position target and integral error

    mujoco.mj_forward(model, data)

    site_pos  = data.site_xpos[site_id].copy()
    error_pos = target - site_pos

    # integral update with anti-windup
    error_integral += error_pos * model.opt.timestep # error accumulation
    error_integral  = np.clip(error_integral, -INTEGRAL_CLIP, INTEGRAL_CLIP)

    control = KP * error_pos + KI * error_integral

    jacp = np.zeros((3, model.nv))
    mujoco.mj_jacSite(model, data, jacp, None, site_id)

    J = jacp[:, ik_dofadr] # only some joints

    A  = J @ J.T + damping * np.eye(3) # pseudo-inverse (3x3)
    dq = J.T @ np.linalg.solve(A, control) # joint update for position control

    q_des_ik += gain * dq

    for i, name in enumerate(ik_joint_names):
        jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)

        if model.jnt_limited[jid]:
            q_min, q_max = model.jnt_range[jid]
            q_des_ik[i]  = np.clip(q_des_ik[i], q_min, q_max)

    return np.linalg.norm(error_pos)

# SIMULATION

with mujoco.viewer.launch_passive(model, data) as viewer:

    for step in range(30000):

        match phase:

            # fase 0: approach handle
            case 0:

                if step % ik_every == 0:

                    dist = solve_position_ik(target_pos, gain_phase0)

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

            # close gripper and wait
            case 1:

                phase_timer += 1

                if phase_timer >= CLOSE_DURATION:
                    mujoco.mj_forward(model, data)

                    data.qvel[:] = 0
                    data.qacc[:] = 0

                    data.eq_active[weld_id] = 1
                    mujoco.mj_forward(model, data)

                    reset_integral()
                    phase = STABILIZE_WELD
                    phase_timer = 0

            # stabilize weld
            case 2:

                phase_timer       += 1
                data.eq_active[weld_id] = 1

                if phase_timer >= 100:
                    handle_index = 1
                    traj_wait_timer = 0
                    reset_integral()
                    phase = OPEN_HANDLE

            # open handle
            case 3:
                '''
                if step % ik_every == 0:

                    data.eq_active[weld_id] = 1

                    target_pos_handle = trajectory_handle[handle_index, 0:3]
                    dist = solve_position_ik(target_pos_handle, gainn_handle)

                    q_des_all[0:5] = q_des_ik
                    q_des_all[5] = joint6_angle
                    q_des_all[6] = joint7_angle

                    if dist < 0.02:
                        traj_wait_timer += 1

                        if traj_wait_timer >= TRAJ_WAIT and handle_index < len(trajectory_handle) - 1:
                            handle_index    += 1
                            traj_wait_timer = 0
                            reset_integral()
                        else:
                            traj_wait_timer = 0

                        if handle_index >= len(trajectory_handle) - 1 and dist < 0.01:
                            traj_index = 1
                            traj_wait_timer = 0
                            reset_integral()
                            phase = OPEN_DOOR
                    '''
                data.eq_active[weld_id] = 1
    
                target_angle = np.deg2rad(-10) * (handle_index / (len(trajectory_handle) - 1))
                data.qpos[handle_qposadr] = target_angle
                data.qpos[door_qposadr]   = 0.0
                mujoco.mj_forward(model, data)
                
                q_des_all[:5] = data.qpos[ik_qposadr]
                q_des_all[5]  = joint6_angle
                q_des_all[6]  = joint7_angle
                
                phase_timer += 1
                if phase_timer % 5 == 0 and handle_index < len(trajectory_handle) - 1:
                    handle_index += 1
                
                if handle_index >= len(trajectory_handle) - 1:
                    traj_index      = 1
                    traj_wait_timer = 0
                    reset_integral()
                    phase = OPEN_DOOR
                

            case 4:

                if step % ik_every == 0:

                    data.eq_active[weld_id] = 1
                    data.qpos[handle_qposadr] = np.deg2rad(-10)

                    target_pos_traj = trajectory[traj_index, 0:3]
                    dist = solve_position_ik(target_pos_traj, gain_phase3)

                    q_des_all[0:5] = q_des_ik
                    q_des_all[5] = joint6_angle
                    q_des_all[6] = joint7_angle

                    if dist < 0.03:
                        traj_wait_timer += 1

                        if traj_wait_timer >= TRAJ_WAIT and traj_index < n_points_open - 1:
                            traj_index += 1
                            traj_wait_timer = 0
                            reset_integral()  # new point, reset integral
                    else:
                        traj_wait_timer = 0

                    if traj_index >= n_points_open - 1 and dist < 0.02:
                        reset_integral()
                        phase = WAIT_OPEN
                        phase_timer = 0

            # door is open, wait before closing
            case 5:

                data.eq_active[weld_id] = 1
                data.qpos[handle_qposadr] = np.deg2rad(-10)
                phase_timer += 1

                if phase_timer >= WAIT_BEFORE_CLOSE_STEPS:
                    data.qvel[:] = 0
                    data.qacc[:] = 0
                    mujoco.mj_forward(model, data)

                    close_index     = 0
                    traj_wait_timer = 0
                    reset_integral()
                    phase = CLOSE_DOOR

            # door closing
            case 6:

                if step % ik_every == 0:

                    data.eq_active[weld_id] = 1
                    data.qpos[handle_qposadr] = np.deg2rad(-10)

                    target_pos_close = trajectory_close[close_index, 0:3]
                    dist = solve_position_ik(target_pos_close, gain_close)

                    q_des_all[0:5] = q_des_ik
                    q_des_all[5]   = joint6_angle
                    q_des_all[6]   = joint7_angle

                    if dist < 0.03:
                        traj_wait_timer += 1

                        if traj_wait_timer >= TRAJ_WAIT and close_index < len(trajectory_close) - 1:
                            close_index    += 1
                            traj_wait_timer = 0
                            reset_integral()   # nuovo waypoint → integrale a zero
                    else:
                        traj_wait_timer = 0

                    if close_index >= len(trajectory_close) - 1 and dist < 0.005:
                        reset_integral()
                        phase = HOLD_CLOSED

            # hold close
            case 7:

                data.eq_active[weld_id] = 1
                phase_timer += 1

                HOLD_STEPS = int(15.0 / model.opt.timestep)  # 3 secondi in steps
                if phase_timer >= HOLD_STEPS:
                    break 

        # actuators
        q_des_all[5] = joint6_angle
        q_des_all[6] = joint7_angle

        data.ctrl[arm_actuators] = q_des_all

        if phase == APPROACH_HANDLE:
            data.ctrl[gripper_actuator] = 255
        else:
            data.ctrl[gripper_actuator] = 50

        mujoco.mj_step(model, data)
        viewer.sync()

        if step % 50 == 0:
            door_angle_deg = np.rad2deg(data.qpos[door_qposadr])
            handle_angle_deg = np.rad2deg(data.qpos[handle_qposadr])
            time_sec = step * model.opt.timestep
            print(f"Tempo {time_sec:.2f} | fase {phase} | maniglia: {handle_angle_deg:.2f} |apertura porta: {door_angle_deg:.2f}°")

    door_angle_deg = np.rad2deg(data.qpos[door_qposadr])
    handle_angle_deg = np.rad2deg(data.qpos[handle_qposadr])
    print(f"Apertura maniglia finale: {handle_angle_deg:.2f}°")
    print(f"Apertura porta finale: {door_angle_deg:.2f}°")

    
'''
    while viewer.is_running():

        data.eq_active[weld_id] = 1

        q_des_all[5] = joint6_angle
        q_des_all[6] = joint7_angle

        data.ctrl[arm_actuators] = q_des_all
        data.ctrl[gripper_actuator] = 50

        mujoco.mj_step(model, data)
        viewer.sync()
'''