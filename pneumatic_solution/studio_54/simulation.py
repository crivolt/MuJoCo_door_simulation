import mujoco
import mujoco.viewer
import numpy as np
import time
import scipy.io


# LOAD MODEL
model = mujoco.MjModel.from_xml_path("/Users/cristianvoltan/Desktop/unipd/tirocinio/CAD/pneumatic_solution/studio_54/studio_54_pneumatic.xml")
data = mujoco.MjData(model)


# DOOR JOINT
door_joint_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "giunto_porta")
door_qposadr  = model.jnt_qposadr[door_joint_id]
door_dofadr   = model.jnt_dofadr[door_joint_id]


# CYLINDER JOINT
cylinder_joint_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "corsa_cilindro")
cylinder_qposadr  = model.jnt_qposadr[cylinder_joint_id]
cylinder_dofadr   = model.jnt_dofadr[cylinder_joint_id]



# ACTUATOR
actuator_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, "forza_pneumatica")

# BASE BALL JOINT OF CYLINDER
base_joint_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "snodo_base_cilindro")
base_qposadr  = model.jnt_qposadr[base_joint_id]
base_dofadr   = model.jnt_dofadr[base_joint_id]

# EQUALITY CONSTRAINT BETWEEN STEM AND HANDLE
connect_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_EQUALITY, "snodo_stelo_maniglia")

if connect_id != -1:
    data.eq_active[connect_id] = 1


# INITIAL FORWARD
mujoco.mj_forward(model, data)


# SIMULATION PARAMETERS
NUM_CYCLES = 1
current_cycle = 0

OPEN_DEG = 62.0
OPEN_RAD = np.deg2rad(OPEN_DEG)

T_OPEN = 1.0
T_WAIT_OPEN = 4.0
T_CLOSE = 1.0

# For quick test use 5 s.
# For standard refrigerated cabinet with one door:
# 10 openings/hour -> one opening every 360 s.
# Total open cycle = 15 s, therefore closed wait = 345 s.
T_WAIT_CLOSED = 5.0

F_MAX = 900.0

K_OPEN = 2000.0
D_OPEN = 150.0

K_CLOSE = 2500.0
D_CLOSE = 120.0

K_HOLD = 700.0
D_HOLD = 150.0

# Studio 54 original frictionloss = 5.402.
# Breakaway friction is modeled as about 10 times the moving friction.
FRICTION_CLOSED = 54.02
FRICTION_MOVING = 5.402


# STATE MACHINE
OPEN_DOOR = 0
WAIT_OPEN = 1
CLOSE_DOOR = 2
WAIT_CLOSED = 3

phase = OPEN_DOOR
phase_start_time = 0.0
phase_start_angle = data.qpos[door_qposadr]


# LOGGING
log_time = []
log_door_deg = []
log_door_vel = []
log_cylinder_pos = []
log_force = []
log_snodo_base_quat = []


# POLYNOMIAL PROFILE
def smoothstep(s):
    s = np.clip(s, 0.0, 1.0)
    return 3.0 * s**2 - 2.0 * s**3


# TEST OPENING DIRECTION
def test_open_direction(force_value):
    test_data = mujoco.MjData(model)

    if connect_id != -1:
        test_data.eq_active[connect_id] = 1

    mujoco.mj_forward(model, test_data)

    for _ in range(500):
        test_data.ctrl[actuator_id] = force_value
        mujoco.mj_step(model, test_data)

    return test_data.qpos[door_qposadr]


theta_positive = test_open_direction(+300.0)
theta_negative = test_open_direction(-300.0)

if theta_positive > theta_negative:
    OPEN_SIGN = +1.0
else:
    OPEN_SIGN = -1.0

print(f"Opening sign selected: {OPEN_SIGN:+.0f}")


# RESET AFTER DIRECTION TEST
data = mujoco.MjData(model)

if connect_id != -1:
    data.eq_active[connect_id] = 1

mujoco.mj_forward(model, data)

phase = OPEN_DOOR
phase_start_time = data.time
phase_start_angle = data.qpos[door_qposadr]


# SIMULATION LOOP
with mujoco.viewer.launch_passive(model, data) as viewer:

    while viewer.is_running() and current_cycle < NUM_CYCLES:

        time_now = data.time
        phase_timer = time_now - phase_start_time

        door_angle = data.qpos[door_qposadr]
        door_vel = data.qvel[door_dofadr]

        # PHASE 0: OPEN DOOR
        if phase == OPEN_DOOR:

            s = smoothstep(phase_timer / T_OPEN)
            theta_des = phase_start_angle + s * (OPEN_RAD - phase_start_angle)

            if door_angle < np.deg2rad(1.0):
                model.dof_frictionloss[door_dofadr] = FRICTION_CLOSED
            else:
                model.dof_frictionloss[door_dofadr] = FRICTION_MOVING

            error = theta_des - door_angle
            force_cmd = OPEN_SIGN * (K_OPEN * error - D_OPEN * door_vel)
            force_cmd = np.clip(force_cmd, -F_MAX, F_MAX)

            if phase_timer >= T_OPEN:
                phase = WAIT_OPEN
                phase_start_time = data.time
                phase_start_angle = data.qpos[door_qposadr]
                print("Phase: WAIT_OPEN")


        # PHASE 1: WAIT WITH DOOR OPEN
        elif phase == WAIT_OPEN:

            theta_des = OPEN_RAD
            error = theta_des - door_angle

            force_cmd = OPEN_SIGN * (K_HOLD * error - D_HOLD * door_vel)
            force_cmd = np.clip(force_cmd, -F_MAX, F_MAX)

            if phase_timer >= T_WAIT_OPEN:
                phase = CLOSE_DOOR
                phase_start_time = data.time
                phase_start_angle = data.qpos[door_qposadr]
                print("Phase: CLOSE_DOOR")


        # PHASE 2: CLOSE DOOR
        elif phase == CLOSE_DOOR:

            s = smoothstep(phase_timer / T_CLOSE)
            theta_des = phase_start_angle + s * (0.0 - phase_start_angle)

            model.dof_frictionloss[door_dofadr] = FRICTION_MOVING

            error = theta_des - door_angle
            force_cmd = OPEN_SIGN * (K_CLOSE * error - D_CLOSE * door_vel)
            force_cmd = np.clip(force_cmd, -F_MAX, F_MAX)

            if phase_timer >= T_CLOSE:
                phase = WAIT_CLOSED
                phase_start_time = data.time
                phase_start_angle = data.qpos[door_qposadr]
                print("Phase: WAIT_CLOSED")


        # PHASE 3: WAIT WITH DOOR CLOSED
        elif phase == WAIT_CLOSED:

            theta_des = 0.0
            error = theta_des - door_angle

            force_cmd = OPEN_SIGN * (K_HOLD * error - D_HOLD * door_vel)
            force_cmd = np.clip(force_cmd, -F_MAX, F_MAX)

            if phase_timer >= T_WAIT_CLOSED:
                current_cycle += 1
                print(f"=== Cycle {current_cycle}/{NUM_CYCLES} completed ===")

                if current_cycle >= NUM_CYCLES:
                    break

                phase = OPEN_DOOR
                phase_start_time = data.time
                phase_start_angle = data.qpos[door_qposadr]
                print("Phase: OPEN_DOOR")


        # SEND CONTROL
        data.ctrl[actuator_id] = force_cmd

        # STEP SIMULATION
        mujoco.mj_step(model, data)
        viewer.sync()

        # SLOW DOWN VISUALIZATION
        time.sleep(model.opt.timestep)


        # LOGGING
        log_time.append(data.time)
        log_door_deg.append(np.rad2deg(data.qpos[door_qposadr]))
        log_door_vel.append(np.rad2deg(data.qvel[door_dofadr]))
        log_cylinder_pos.append(data.qpos[cylinder_qposadr])
        log_force.append(force_cmd)
        log_snodo_base_quat.append(data.qpos[base_qposadr:base_qposadr + 4].copy())


        # PRINT DIAGNOSTICS
        if len(log_time) % 50 == 0:
            print(
                f"Time {data.time:.2f} | phase {phase} | "
                f"door: {np.rad2deg(data.qpos[door_qposadr]):.2f}° | "
                f"cylinder: {data.qpos[cylinder_qposadr]:.3f} m | "
                f"force: {force_cmd:.2f} N"
            )


# FINAL OUTPUT
door_angle_deg = np.rad2deg(data.qpos[door_qposadr])
print(f"Final door angle: {door_angle_deg:.2f}°")


# SAVE DATA WITHOUT SCIPY
scipy.io.savemat("dati_porta_pneumatica_studio_54.mat", {
    "tempo": np.array(log_time),
    "apertura": np.array(log_door_deg),
    "velocita": np.array(log_door_vel),
    "posizione_cilindro": np.array(log_cylinder_pos),
    "forza": np.array(log_force),
    "quat_snodo_base": np.array(log_snodo_base_quat),
})