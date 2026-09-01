import mujoco
import mujoco.viewer
import numpy as np
import time
import scipy.io


# LOAD MODEL
MODEL_PATH = "/Users/cristianvoltan/Desktop/unipd/tirocinio/CAD/electric_solution/electrolux/electrolux_ele.xml"
model = mujoco.MjModel.from_xml_path(MODEL_PATH)
data = mujoco.MjData(model)


def get_joint_addr(model, name):
    jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)
    return jid, model.jnt_qposadr[jid], model.jnt_dofadr[jid]


def get_actuator_id(model, name):
    aid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, name)
    return aid


# JOINTS
door_joint_id, door_qposadr, door_dofadr = get_joint_addr(model, "giunto_porta")
motor_joint_id, motor_qposadr, motor_dofadr = get_joint_addr(model, "giunto_rotante")
comp_radiale_joint_id, comp_radiale_qposadr, comp_radiale_dofadr = get_joint_addr(model, "compensazione_radiale")
align_joint_id, align_qposadr, align_dofadr = get_joint_addr(model, "snodo_allineamento")

# ACTUATORS
motor_actuator_id = get_actuator_id(model, "servo_motoriduttore")

# EQUALITY CONSTRAINT BETWEEN END EFFECTOR AND HANDLE
connect_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_EQUALITY, "presa_meccanica_maniglia")
data.eq_active[connect_id] = 1

mujoco.mj_forward(model, data)


# SIMULATION PARAMETERS
NUM_CYCLES = 1
current_cycle = 0

OPEN_DEG = 62.0
OPEN_RAD = np.deg2rad(OPEN_DEG)

T_OPEN = 1.0
T_WAIT_OPEN = 4.0
T_CLOSE = 1.0
T_WAIT_CLOSED = 5.0

TORQUE_MAX = 150.0

K_OPEN = 2500
D_OPEN = 120

K_CLOSE = 2500
D_CLOSE = 120

K_HOLD = 2000
D_HOLD = 150

FRICTION_CLOSED = 62.06
FRICTION_MOVING = 6.206

# STATE MACHINE
OPEN_DOOR = 0
WAIT_OPEN = 1
CLOSE_DOOR = 2
WAIT_CLOSED = 3

phase = OPEN_DOOR
phase_timer = 0.0
phase_start_time = 0.0
phase_start_angle = data.qpos[door_qposadr]


# LOGGING 
log_time = []
log_door_deg = []
log_door_vel_deg = []
log_motor_deg = []  
log_motor_vel_deg = []
log_motor_torque_cmd = []
log_motor_torque_real = []
log_comp_radiale_mm = []
log_align_deg = []
log_motor_power = []


def quat_angle_deg(quat):
    # angolo di rotazione (in gradi) rispetto al quaternione identita' (1,0,0,0)
    w = np.clip(quat[0], -1.0, 1.0)
    return np.rad2deg(2.0 * np.arccos(np.abs(w)))


def smoothstep(s):
    s = np.clip(s, 0.0, 1.0)
    return 3.0 * s**2 - 2.0 * s**3


def test_open_direction(test_torque):
    test_data = mujoco.MjData(model)
    test_data.eq_active[connect_id] = 1
    mujoco.mj_forward(model, test_data)

    for _ in range(800):
        test_data.ctrl[motor_actuator_id] = test_torque
        mujoco.mj_step(model, test_data)

    return test_data.qpos[door_qposadr]


theta_positive = test_open_direction(+50.0)
theta_negative = test_open_direction(-50.0)
OPEN_SIGN = 1.0 if theta_positive > theta_negative else -1.0
print(f"Opening sign selected: {OPEN_SIGN:+.0f}")


# RESET AFTER DIRECTION TEST
data = mujoco.MjData(model)
data.eq_active[connect_id] = 1
mujoco.mj_forward(model, data)

phase = OPEN_DOOR
phase_start_time = data.time
phase_start_angle = data.qpos[door_qposadr]

motor_ctrl = 0.0


# SIMULATION LOOP
with mujoco.viewer.launch_passive(model, data) as viewer:

    while viewer.is_running() and current_cycle < NUM_CYCLES:

        time_now = data.time
        phase_timer = time_now - phase_start_time

        door_angle = data.qpos[door_qposadr]
        door_vel = data.qvel[door_dofadr]
        motor_angle = data.qpos[motor_qposadr]

        # PHASE 0: OPEN DOOR
        if phase == OPEN_DOOR:
            s = smoothstep(phase_timer / T_OPEN)
            theta_des = phase_start_angle + s * (OPEN_RAD - phase_start_angle)

            model.dof_frictionloss[door_dofadr] = (FRICTION_CLOSED if door_angle < np.deg2rad(1.0) else FRICTION_MOVING)

            error = theta_des - door_angle
            tau_motor = OPEN_SIGN * (K_OPEN * error - D_OPEN * door_vel)
            motor_ctrl = np.clip(tau_motor, -TORQUE_MAX, TORQUE_MAX)

            if phase_timer >= T_OPEN:
                phase = WAIT_OPEN
                phase_start_time = data.time
                phase_start_angle = data.qpos[door_qposadr]
                print("Phase: WAIT_OPEN")

        # PHASE 1: WAIT WITH DOOR OPEN
        elif phase == WAIT_OPEN:
            error = OPEN_RAD - door_angle
            tau_motor = OPEN_SIGN * (K_HOLD * error - D_HOLD * door_vel)
            motor_ctrl = np.clip(tau_motor, -TORQUE_MAX, TORQUE_MAX)

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
            tau_motor = OPEN_SIGN * (K_CLOSE * error - D_CLOSE * door_vel)
            motor_ctrl = np.clip(tau_motor, -TORQUE_MAX, TORQUE_MAX)

            if phase_timer >= T_CLOSE:
                phase = WAIT_CLOSED
                phase_start_time = data.time
                phase_start_angle = data.qpos[door_qposadr]
                print("Phase: WAIT_CLOSED")

        # PHASE 3: WAIT WITH DOOR CLOSED
        elif phase == WAIT_CLOSED:
            # ripristina l'attrito statico ora che la porta e' ferma e chiusa
            model.dof_frictionloss[door_dofadr] = FRICTION_CLOSED

            error = 0.0 - door_angle
            tau_motor = OPEN_SIGN * (K_HOLD * error - D_HOLD * door_vel)
            motor_ctrl = np.clip(tau_motor, -TORQUE_MAX, TORQUE_MAX)
            

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
        data.ctrl[motor_actuator_id] = motor_ctrl

        # STEP SIMULATION
        mujoco.mj_step(model, data)
        viewer.sync()
        time.sleep(model.opt.timestep)

        # MOTOR TORQUE AND POWER
        motor_vel_rad = data.qvel[motor_dofadr]
        motor_torque_real = data.qfrc_actuator[motor_dofadr]
        motor_power = motor_torque_real * motor_vel_rad

        # LOGGING
        log_time.append(data.time)
        log_door_deg.append(np.rad2deg(data.qpos[door_qposadr]))
        log_door_vel_deg.append(np.rad2deg(data.qvel[door_dofadr]))
        log_motor_deg.append(np.rad2deg(data.qpos[motor_qposadr]))
        log_motor_vel_deg.append(np.rad2deg(data.qvel[motor_dofadr]))
        log_motor_torque_cmd.append(motor_ctrl)
        log_motor_torque_real.append(motor_torque_real)
        log_motor_power.append(motor_power)
        log_comp_radiale_mm.append(data.qpos[comp_radiale_qposadr] * 1000.0)
        log_align_deg.append(quat_angle_deg(data.qpos[align_qposadr:align_qposadr + 4]))

        # PRINT DIAGNOSTICS
        if len(log_time) % 50 == 0:
            print(
                f"Time {data.time:.2f} | phase {phase} | "
                f"door: {np.rad2deg(data.qpos[door_qposadr]):.2f} deg | "
                f"motor: {np.rad2deg(data.qpos[motor_qposadr]):.2f} deg | "
                f"comp_radiale: {data.qpos[comp_radiale_qposadr] * 1000:.2f} mm | "
                f"align: {log_align_deg[-1]:.2f} deg"
            )


# FINAL OUTPUT
door_angle_deg = np.rad2deg(data.qpos[door_qposadr])
print(f"Final door angle: {door_angle_deg:.2f} deg")


scipy.io.savemat("dati_porta_meccanica_electrolux.mat", {
    "tempo": np.array(log_time),
    "posizione_porta": np.array(log_door_deg),
    "velocita_porta": np.array(log_door_vel_deg),
    "posizione_motoriduttore": np.array(log_motor_deg),
    "velocita_motoriduttore": np.array(log_motor_vel_deg),
    "correzione_allineamento_radiale_mm": np.array(log_comp_radiale_mm),
    "rotazione_allineamento_deg": np.array(log_align_deg),
    "coppia_motoriduttore_cmd": np.array(log_motor_torque_cmd),
    "coppia_motoriduttore": np.array(log_motor_torque_real),
    "potenza_meccanica": np.array(log_motor_power),
})