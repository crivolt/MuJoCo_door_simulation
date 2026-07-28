import mujoco
import mujoco.viewer
import numpy as np
import time
import scipy.io


# LOAD MODEL
MODEL_PATH = "/Users/cristianvoltan/Desktop/unipd/tirocinio/CAD/electric_solution/osaka_3p/osaka_ele.xml"

model = mujoco.MjModel.from_xml_path(MODEL_PATH)
data = mujoco.MjData(model)


def get_joint_addr(model, name):
    jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)
    return jid, model.jnt_qposadr[jid], model.jnt_dofadr[jid]


def get_actuator_id(model, name):
    aid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, name)
    return aid


def get_equality_id(model, name):
    eid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_EQUALITY, name)
    return eid


def quat_angle_deg(quat):
    # Angolo di rotazione in gradi rispetto al quaternione identità
    w = np.clip(quat[0], -1.0, 1.0)
    return np.rad2deg(2.0 * np.arccos(np.abs(w)))


def smoothstep(s):
    s = np.clip(s, 0.0, 1.0)
    return 3.0 * s**2 - 2.0 * s**3


# JOINTS PORTA SINISTRA
door_sx_joint_id, door_sx_qposadr, door_sx_dofadr = get_joint_addr(model, "giunto_porta_sx")
motor_sx_joint_id, motor_sx_qposadr, motor_sx_dofadr = get_joint_addr(model, "giunto_rotante_sx")
comp_sx_joint_id, comp_sx_qposadr, comp_sx_dofadr = get_joint_addr(model, "compensazione_radiale_sx")
align_sx_joint_id, align_sx_qposadr, align_sx_dofadr = get_joint_addr(model, "snodo_allineamento_sx")


# JOINTS PORTA DESTRA
door_dx_joint_id, door_dx_qposadr, door_dx_dofadr = get_joint_addr(model, "giunto_porta_dx")
motor_dx_joint_id, motor_dx_qposadr, motor_dx_dofadr = get_joint_addr(model, "giunto_rotante_dx")
comp_dx_joint_id, comp_dx_qposadr, comp_dx_dofadr = get_joint_addr(model, "compensazione_radiale_dx")
align_dx_joint_id, align_dx_qposadr, align_dx_dofadr = get_joint_addr(model, "snodo_allineamento_dx")


# ACTUATORS
motor_sx_actuator_id = get_actuator_id(model, "motore_motoriduttore_sx")
motor_dx_actuator_id = get_actuator_id(model, "motore_motoriduttore_dx")


# EQUALITY CONSTRAINTS BETWEEN END EFFECTORS AND HANDLES
connect_sx_id = get_equality_id(model, "presa_meccanica_maniglia_sx")
connect_dx_id = get_equality_id(model, "presa_meccanica_maniglia_dx")

data.eq_active[connect_sx_id] = 1
data.eq_active[connect_dx_id] = 1

mujoco.mj_forward(model, data)


# SIMULATION PARAMETERS
OPEN_DEG = 62.0
OPEN_RAD_DX = np.deg2rad(OPEN_DEG)
OPEN_RAD_SX = -np.deg2rad(OPEN_DEG)

T_OPEN = 1.0
T_WAIT_OPEN = 13.0
T_CLOSE = 1.0
T_WAIT_CLOSED = 2.0

TORQUE_MAX = 150.0

K_OPEN = 2500.0
D_OPEN = 120.0

K_CLOSE = 3000.0
D_CLOSE = 100.0

K_HOLD = 2000.0
D_HOLD = 150.0

FRICTION_CLOSED = 5.28
FRICTION_MOVING = 1.056


# STATE MACHINE
OPEN_SX = 0
WAIT_OPEN_SX = 1
CLOSE_SX = 2
WAIT_CLOSED_SX = 3
OPEN_DX = 4
WAIT_OPEN_DX = 5
CLOSE_DX = 6
WAIT_CLOSED_DX = 7

phase = OPEN_SX
phase_start_time = 0.0
phase_start_angle_sx = data.qpos[door_sx_qposadr]
phase_start_angle_dx = data.qpos[door_dx_qposadr]


# LOGGING
log_time = []

log_door_sx_deg = []
log_door_sx_vel_deg = []
log_motor_sx_deg = []
log_comp_sx_mm = []
log_align_sx_deg = []

log_door_dx_deg = []
log_door_dx_vel_deg = []
log_motor_dx_deg = []
log_comp_dx_mm = []
log_align_dx_deg = []


def test_motor_to_door_direction(motor_actuator_id, door_qposadr, test_torque):
    # Applica una coppia di test al motoriduttore e misura come si muove la porta associata
    test_data = mujoco.MjData(model)
    test_data.eq_active[connect_sx_id] = 1
    test_data.eq_active[connect_dx_id] = 1
    mujoco.mj_forward(model, test_data)

    for _ in range(800):
        test_data.ctrl[:] = 0.0
        test_data.ctrl[motor_actuator_id] = test_torque
        mujoco.mj_step(model, test_data)

    return test_data.qpos[door_qposadr]


# TEST AUTOMATICO DEL VERSO DEI DUE MOTORI
theta_sx_positive = test_motor_to_door_direction(motor_sx_actuator_id, door_sx_qposadr, +50.0)
theta_sx_negative = test_motor_to_door_direction(motor_sx_actuator_id, door_sx_qposadr, -50.0)

theta_dx_positive = test_motor_to_door_direction(motor_dx_actuator_id, door_dx_qposadr, +50.0)
theta_dx_negative = test_motor_to_door_direction(motor_dx_actuator_id, door_dx_qposadr, -50.0)

MOTOR_TO_DOOR_SIGN_SX = 1.0 if theta_sx_positive > theta_sx_negative else -1.0
MOTOR_TO_DOOR_SIGN_DX = 1.0 if theta_dx_positive > theta_dx_negative else -1.0

print(f"Door SX angle after +50 Nm test: {np.rad2deg(theta_sx_positive):.3f} deg")
print(f"Door SX angle after -50 Nm test: {np.rad2deg(theta_sx_negative):.3f} deg")
print(f"Motor-to-door sign SX selected: {MOTOR_TO_DOOR_SIGN_SX:+.0f}")

print(f"Door DX angle after +50 Nm test: {np.rad2deg(theta_dx_positive):.3f} deg")
print(f"Door DX angle after -50 Nm test: {np.rad2deg(theta_dx_negative):.3f} deg")
print(f"Motor-to-door sign DX selected: {MOTOR_TO_DOOR_SIGN_DX:+.0f}")


# RESET AFTER DIRECTION TEST
data = mujoco.MjData(model)
data.eq_active[connect_sx_id] = 1
data.eq_active[connect_dx_id] = 1
mujoco.mj_forward(model, data)

phase = OPEN_SX
phase_start_time = data.time
phase_start_angle_sx = data.qpos[door_sx_qposadr]
phase_start_angle_dx = data.qpos[door_dx_qposadr]

motor_ctrl_sx = 0.0
motor_ctrl_dx = 0.0


def hold_closed_torque(sign, door_angle, door_vel):
    # Tiene la porta chiusa a 0 rad
    error = 0.0 - door_angle
    tau = sign * (K_HOLD * error - D_HOLD * door_vel)
    return np.clip(tau, -TORQUE_MAX, TORQUE_MAX)


# SIMULATION LOOP
with mujoco.viewer.launch_passive(model, data) as viewer:

    while viewer.is_running():

        time_now = data.time
        phase_timer = time_now - phase_start_time

        door_sx_angle = data.qpos[door_sx_qposadr]
        door_sx_vel = data.qvel[door_sx_dofadr]

        door_dx_angle = data.qpos[door_dx_qposadr]
        door_dx_vel = data.qvel[door_dx_dofadr]

        # PHASE 0: OPEN LEFT DOOR
        if phase == OPEN_SX:

            s = smoothstep(phase_timer / T_OPEN)
            theta_des_sx = phase_start_angle_sx + s * (OPEN_RAD_SX - phase_start_angle_sx)

            model.dof_frictionloss[door_sx_dofadr] = (FRICTION_CLOSED if abs(door_sx_angle) < np.deg2rad(1.0) else FRICTION_MOVING)
            model.dof_frictionloss[door_dx_dofadr] = FRICTION_CLOSED

            error_sx = theta_des_sx - door_sx_angle

            tau_sx = MOTOR_TO_DOOR_SIGN_SX * (K_OPEN * error_sx - D_OPEN * door_sx_vel)
            motor_ctrl_sx = np.clip(tau_sx, -TORQUE_MAX, TORQUE_MAX)

            # motor_ctrl_dx = hold_closed_torque(MOTOR_TO_DOOR_SIGN_DX, door_dx_angle, door_dx_vel)

            if phase_timer >= T_OPEN:
                phase = WAIT_OPEN_SX
                phase_start_time = data.time
                phase_start_angle_sx = data.qpos[door_sx_qposadr]
                print("Phase: WAIT_OPEN_SX")

        # PHASE 1: WAIT WITH LEFT DOOR OPEN
        elif phase == WAIT_OPEN_SX:

            error_sx = OPEN_RAD_SX - door_sx_angle

            tau_sx = MOTOR_TO_DOOR_SIGN_SX * (K_HOLD * error_sx - D_HOLD * door_sx_vel)
            motor_ctrl_sx = np.clip(tau_sx, -TORQUE_MAX, TORQUE_MAX)

            # motor_ctrl_dx = hold_closed_torque(MOTOR_TO_DOOR_SIGN_DX, door_dx_angle, door_dx_vel)

            if phase_timer >= T_WAIT_OPEN:
                phase = CLOSE_SX
                phase_start_time = data.time
                phase_start_angle_sx = data.qpos[door_sx_qposadr]
                print("Phase: CLOSE_SX")

        # PHASE 2: CLOSE LEFT DOOR
        elif phase == CLOSE_SX:

            s = smoothstep(phase_timer / T_CLOSE)
            theta_des_sx = phase_start_angle_sx + s * (0.0 - phase_start_angle_sx)

            model.dof_frictionloss[door_sx_dofadr] = FRICTION_MOVING
            model.dof_frictionloss[door_dx_dofadr] = FRICTION_CLOSED

            error_sx = theta_des_sx - door_sx_angle

            tau_sx = MOTOR_TO_DOOR_SIGN_SX * (K_CLOSE * error_sx - D_CLOSE * door_sx_vel)
            motor_ctrl_sx = np.clip(tau_sx, -TORQUE_MAX, TORQUE_MAX)

            # motor_ctrl_dx = hold_closed_torque(MOTOR_TO_DOOR_SIGN_DX, door_dx_angle, door_dx_vel)

            if phase_timer >= T_CLOSE:
                phase = WAIT_CLOSED_SX
                phase_start_time = data.time
                phase_start_angle_sx = data.qpos[door_sx_qposadr]
                print("Phase: WAIT_CLOSED_SX")

        # PHASE 3: WAIT WITH LEFT DOOR CLOSED
        elif phase == WAIT_CLOSED_SX:

            model.dof_frictionloss[door_sx_dofadr] = FRICTION_CLOSED
            model.dof_frictionloss[door_dx_dofadr] = FRICTION_CLOSED

            motor_ctrl_sx = hold_closed_torque(MOTOR_TO_DOOR_SIGN_SX, door_sx_angle, door_sx_vel)
            # motor_ctrl_dx = hold_closed_torque(MOTOR_TO_DOOR_SIGN_DX, door_dx_angle, door_dx_vel)

            if phase_timer >= T_WAIT_CLOSED:
                phase = OPEN_DX
                phase_start_time = data.time
                phase_start_angle_dx = data.qpos[door_dx_qposadr]
                print("Phase: OPEN_DX")

        # PHASE 4: OPEN RIGHT DOOR
        elif phase == OPEN_DX:

            s = smoothstep(phase_timer / T_OPEN)
            theta_des_dx = phase_start_angle_dx + s * (OPEN_RAD_DX - phase_start_angle_dx)

            model.dof_frictionloss[door_sx_dofadr] = FRICTION_CLOSED
            model.dof_frictionloss[door_dx_dofadr] = (FRICTION_CLOSED if abs(door_dx_angle) < np.deg2rad(1.0) else FRICTION_MOVING)

            # motor_ctrl_sx = hold_closed_torque(MOTOR_TO_DOOR_SIGN_SX, door_sx_angle, door_sx_vel)

            error_dx = theta_des_dx - door_dx_angle

            tau_dx = MOTOR_TO_DOOR_SIGN_DX * (K_OPEN * error_dx - D_OPEN * door_dx_vel)
            motor_ctrl_dx = np.clip(tau_dx, -TORQUE_MAX, TORQUE_MAX)

            if phase_timer >= T_OPEN:
                phase = WAIT_OPEN_DX
                phase_start_time = data.time
                phase_start_angle_dx = data.qpos[door_dx_qposadr]
                print("Phase: WAIT_OPEN_DX")

        # PHASE 5: WAIT WITH RIGHT DOOR OPEN
        elif phase == WAIT_OPEN_DX:

            # motor_ctrl_sx = hold_closed_torque(MOTOR_TO_DOOR_SIGN_SX, door_sx_angle, door_sx_vel)

            error_dx = OPEN_RAD_DX - door_dx_angle

            tau_dx = MOTOR_TO_DOOR_SIGN_DX * (K_HOLD * error_dx - D_HOLD * door_dx_vel)
            motor_ctrl_dx = np.clip(tau_dx, -TORQUE_MAX, TORQUE_MAX)

            if phase_timer >= T_WAIT_OPEN:
                phase = CLOSE_DX
                phase_start_time = data.time
                phase_start_angle_dx = data.qpos[door_dx_qposadr]
                print("Phase: CLOSE_DX")

        # PHASE 6: CLOSE RIGHT DOOR
        elif phase == CLOSE_DX:

            s = smoothstep(phase_timer / T_CLOSE)
            theta_des_dx = phase_start_angle_dx + s * (0.0 - phase_start_angle_dx)

            model.dof_frictionloss[door_sx_dofadr] = FRICTION_CLOSED
            model.dof_frictionloss[door_dx_dofadr] = FRICTION_MOVING

            # motor_ctrl_sx = hold_closed_torque(MOTOR_TO_DOOR_SIGN_SX, door_sx_angle, door_sx_vel)

            error_dx = theta_des_dx - door_dx_angle

            tau_dx = MOTOR_TO_DOOR_SIGN_DX * (K_CLOSE * error_dx - D_CLOSE * door_dx_vel)
            motor_ctrl_dx = np.clip(tau_dx, -TORQUE_MAX, TORQUE_MAX)

            if phase_timer >= T_CLOSE:
                phase = WAIT_CLOSED_DX
                phase_start_time = data.time
                phase_start_angle_dx = data.qpos[door_dx_qposadr]
                print("Phase: WAIT_CLOSED_DX")

        # PHASE 7: WAIT WITH RIGHT DOOR CLOSED
        elif phase == WAIT_CLOSED_DX:

            model.dof_frictionloss[door_sx_dofadr] = FRICTION_CLOSED
            model.dof_frictionloss[door_dx_dofadr] = FRICTION_CLOSED

            # motor_ctrl_sx = hold_closed_torque(MOTOR_TO_DOOR_SIGN_SX, door_sx_angle, door_sx_vel)
            motor_ctrl_dx = hold_closed_torque(MOTOR_TO_DOOR_SIGN_DX, door_dx_angle, door_dx_vel)

            if phase_timer >= T_WAIT_CLOSED:
                print("=== Sequence completed: SX open/close + DX open/close ===")
                break

        # SEND CONTROL
        data.ctrl[motor_sx_actuator_id] = motor_ctrl_sx
        data.ctrl[motor_dx_actuator_id] = motor_ctrl_dx

        # STEP SIMULATION
        mujoco.mj_step(model, data)
        viewer.sync()
        time.sleep(model.opt.timestep)

        # LOGGING
        log_time.append(data.time)

        log_door_sx_deg.append(np.rad2deg(data.qpos[door_sx_qposadr]))
        log_door_sx_vel_deg.append(np.rad2deg(data.qvel[door_sx_dofadr]))
        log_motor_sx_deg.append(np.rad2deg(data.qpos[motor_sx_qposadr]))
        log_comp_sx_mm.append(data.qpos[comp_sx_qposadr] * 1000.0)
        log_align_sx_deg.append(quat_angle_deg(data.qpos[align_sx_qposadr:align_sx_qposadr + 4]))

        log_door_dx_deg.append(np.rad2deg(data.qpos[door_dx_qposadr]))
        log_door_dx_vel_deg.append(np.rad2deg(data.qvel[door_dx_dofadr]))
        log_motor_dx_deg.append(np.rad2deg(data.qpos[motor_dx_qposadr]))
        log_comp_dx_mm.append(data.qpos[comp_dx_qposadr] * 1000.0)
        log_align_dx_deg.append(quat_angle_deg(data.qpos[align_dx_qposadr:align_dx_qposadr + 4]))

        # PRINT DIAGNOSTICS
        if len(log_time) % 50 == 0:
            print(
                f"Time {data.time:.2f} | phase {phase} | "
                f"porta_sx: {np.rad2deg(data.qpos[door_sx_qposadr]):.2f} deg | "
                f"motore_sx: {np.rad2deg(data.qpos[motor_sx_qposadr]):.2f} deg | "
                f"comp_sx: {data.qpos[comp_sx_qposadr] * 1000:.2f} mm | "
                f"align_sx: {log_align_sx_deg[-1]:.2f} deg | "
                f"porta_dx: {np.rad2deg(data.qpos[door_dx_qposadr]):.2f} deg | "
                f"motore_dx: {np.rad2deg(data.qpos[motor_dx_qposadr]):.2f} deg | "
                f"comp_dx: {data.qpos[comp_dx_qposadr] * 1000:.2f} mm | "
                f"align_dx: {log_align_dx_deg[-1]:.2f} deg"
            )


# FINAL OUTPUT
door_sx_angle_deg = np.rad2deg(data.qpos[door_sx_qposadr])
door_dx_angle_deg = np.rad2deg(data.qpos[door_dx_qposadr])

print(f"Final door SX angle: {door_sx_angle_deg:.2f} deg")
print(f"Final door DX angle: {door_dx_angle_deg:.2f} deg")


# SAVE DATA
scipy.io.savemat("dati_porta_meccanica_astana_p.mat", {
    "tempo": np.array(log_time),

    "posizione_porta_sx": np.array(log_door_sx_deg),
    "velocita_porta_sx": np.array(log_door_sx_vel_deg),
    "posizione_motoriduttore_sx": np.array(log_motor_sx_deg),
    "correzione_allineamento_radiale_sx_mm": np.array(log_comp_sx_mm),
    "rotazione_allineamento_sx_deg": np.array(log_align_sx_deg),

    "posizione_porta_dx": np.array(log_door_dx_deg),
    "velocita_porta_dx": np.array(log_door_dx_vel_deg),
    "posizione_motoriduttore_dx": np.array(log_motor_dx_deg),
    "correzione_allineamento_radiale_dx_mm": np.array(log_comp_dx_mm),
    "rotazione_allineamento_dx_deg": np.array(log_align_dx_deg),
})

print("Dati salvati in: dati_porta_meccanica_astana_p.mat")