import mujoco
import mujoco.viewer
import numpy as np
import time
import scipy.io


# LOAD MODEL
MODEL_PATH = "/Users/cristianvoltan/Desktop/unipd/tirocinio/CAD/electric_solution/osaka/osaka_ele.xml"

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
    w = np.clip(quat[0], -1.0, 1.0)
    return np.rad2deg(2.0 * np.arccos(np.abs(w)))


def smoothstep(s):
    s = np.clip(s, 0.0, 1.0)
    return 3.0 * s**2 - 2.0 * s**3


# JOINTS PORTA SINISTRA
door_sx_joint_id, door_sx_qposadr, door_sx_dofadr = get_joint_addr(model, "slide_sx")
act_sx_joint_id, act_sx_qposadr, act_sx_dofadr = get_joint_addr(model, "attuatore_lineare_sx")
align_sx_joint_id, align_sx_qposadr, align_sx_dofadr = get_joint_addr(model, "snodo_allineamento_lineare_sx")


# JOINTS PORTA DESTRA
door_dx_joint_id, door_dx_qposadr, door_dx_dofadr = get_joint_addr(model, "slide_dx")
act_dx_joint_id, act_dx_qposadr, act_dx_dofadr = get_joint_addr(model, "attuatore_lineare_dx")
align_dx_joint_id, align_dx_qposadr, align_dx_dofadr = get_joint_addr(model, "snodo_allineamento_lineare_dx")


# ACTUATORS
motor_sx_actuator_id = get_actuator_id(model, "motore_lineare_sx")
motor_dx_actuator_id = get_actuator_id(model, "motore_lineare_dx")


# EQUALITY CONSTRAINTS BETWEEN END EFFECTORS AND HANDLES
connect_sx_id = get_equality_id(model, "presa_lineare_maniglia_sx")
connect_dx_id = get_equality_id(model, "presa_lineare_maniglia_dx")

data.eq_active[connect_sx_id] = 1
data.eq_active[connect_dx_id] = 1

mujoco.mj_forward(model, data)


# SIMULATION PARAMETERS
OPEN_DISTANCE = 0.7

T_OPEN = 1.0
T_WAIT_OPEN = 13.0
T_CLOSE = 1.0
T_WAIT_CLOSED = 2.0

FORCE_MAX = 800.0

K_OPEN = 2000.0
D_OPEN = 120.0

K_CLOSE = 2000.0
D_CLOSE = 120.0

K_HOLD = 2000.0
D_HOLD = 150.0

FRICTION_CLOSED = 33.0
FRICTION_MOVING = 3.0


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
phase_start_pos_sx = data.qpos[door_sx_qposadr]
phase_start_pos_dx = data.qpos[door_dx_qposadr]


# LOGGING
log_time = []

log_door_sx_m = []
log_door_sx_vel_m_s = []
log_act_sx_m = []
log_act_sx_vel_m_s = []
log_align_sx_deg = []

log_door_dx_m = []
log_door_dx_vel_m_s = []
log_act_dx_m = []
log_act_dx_vel_m_s = []
log_align_dx_deg = []

log_force_sx = []
log_force_dx = []

log_power_sx = []
log_power_dx = []


def test_actuator_to_door_direction(motor_actuator_id, door_qposadr, test_force):
    test_data = mujoco.MjData(model)
    test_data.eq_active[connect_sx_id] = 1
    test_data.eq_active[connect_dx_id] = 1
    mujoco.mj_forward(model, test_data)

    for _ in range(800):
        test_data.ctrl[:] = 0.0
        test_data.ctrl[motor_actuator_id] = test_force
        mujoco.mj_step(model, test_data)

    return test_data.qpos[door_qposadr]


# TEST AUTOMATICO DEL VERSO DEI DUE ATTUATORI
x_sx_positive = test_actuator_to_door_direction(motor_sx_actuator_id, door_sx_qposadr, +200.0)
x_sx_negative = test_actuator_to_door_direction(motor_sx_actuator_id, door_sx_qposadr, -200.0)

x_dx_positive = test_actuator_to_door_direction(motor_dx_actuator_id, door_dx_qposadr, +200.0)
x_dx_negative = test_actuator_to_door_direction(motor_dx_actuator_id, door_dx_qposadr, -200.0)

ACT_TO_DOOR_SIGN_SX = 1.0 if x_sx_positive > x_sx_negative else -1.0
ACT_TO_DOOR_SIGN_DX = 1.0 if x_dx_positive > x_dx_negative else -1.0

print(f"Door SX position after +200 N test: {x_sx_positive * 1000:.2f} mm")
print(f"Door SX position after -200 N test: {x_sx_negative * 1000:.2f} mm")
print(f"Actuator-to-door sign SX selected: {ACT_TO_DOOR_SIGN_SX:+.0f}")

print(f"Door DX position after +200 N test: {x_dx_positive * 1000:.2f} mm")
print(f"Door DX position after -200 N test: {x_dx_negative * 1000:.2f} mm")
print(f"Actuator-to-door sign DX selected: {ACT_TO_DOOR_SIGN_DX:+.0f}")


# RESET AFTER DIRECTION TEST
data = mujoco.MjData(model)
data.eq_active[connect_sx_id] = 1
data.eq_active[connect_dx_id] = 1
mujoco.mj_forward(model, data)

phase = OPEN_SX
phase_start_time = data.time
phase_start_pos_sx = data.qpos[door_sx_qposadr]
phase_start_pos_dx = data.qpos[door_dx_qposadr]

motor_ctrl_sx = 0.0
motor_ctrl_dx = 0.0


def hold_closed_force(sign, door_pos, door_vel):
    error = 0.0 - door_pos
    force = sign * (K_HOLD * error - D_HOLD * door_vel)
    return np.clip(force, -FORCE_MAX, FORCE_MAX)


# SIMULATION LOOP
with mujoco.viewer.launch_passive(model, data) as viewer:

    while viewer.is_running():

        time_now = data.time
        phase_timer = time_now - phase_start_time

        door_sx_pos = data.qpos[door_sx_qposadr]
        door_sx_vel = data.qvel[door_sx_dofadr]

        door_dx_pos = data.qpos[door_dx_qposadr]
        door_dx_vel = data.qvel[door_dx_dofadr]

        motor_ctrl_sx = 0.0
        motor_ctrl_dx = 0.0     

        # PHASE 0: OPEN LEFT DOOR
        if phase == OPEN_SX:

            s = smoothstep(phase_timer / T_OPEN)
            x_des_sx = phase_start_pos_sx + s * (OPEN_DISTANCE - phase_start_pos_sx)

            model.dof_frictionloss[door_sx_dofadr] = FRICTION_CLOSED if door_sx_pos < 0.001 else FRICTION_MOVING
            model.dof_frictionloss[door_dx_dofadr] = FRICTION_CLOSED

            error_sx = x_des_sx - door_sx_pos

            force_sx = ACT_TO_DOOR_SIGN_SX * (K_OPEN * error_sx - D_OPEN * door_sx_vel)
            motor_ctrl_sx = np.clip(force_sx, -FORCE_MAX, FORCE_MAX)

            # motor_ctrl_dx = hold_closed_force(ACT_TO_DOOR_SIGN_DX, door_dx_pos, door_dx_vel)

            if phase_timer >= T_OPEN:
                phase = WAIT_OPEN_SX
                phase_start_time = data.time
                phase_start_pos_sx = data.qpos[door_sx_qposadr]
                print("Phase: WAIT_OPEN_SX")

        # PHASE 1: WAIT WITH LEFT DOOR OPEN
        elif phase == WAIT_OPEN_SX:

            error_sx = OPEN_DISTANCE - door_sx_pos

            force_sx = ACT_TO_DOOR_SIGN_SX * (K_HOLD * error_sx - D_HOLD * door_sx_vel)
            motor_ctrl_sx = np.clip(force_sx, -FORCE_MAX, FORCE_MAX)

            # motor_ctrl_dx = hold_closed_force(ACT_TO_DOOR_SIGN_DX, door_dx_pos, door_dx_vel)

            if phase_timer >= T_WAIT_OPEN:
                phase = CLOSE_SX
                phase_start_time = data.time
                phase_start_pos_sx = data.qpos[door_sx_qposadr]
                print("Phase: CLOSE_SX")

        # PHASE 2: CLOSE LEFT DOOR
        elif phase == CLOSE_SX:

            s = smoothstep(phase_timer / T_CLOSE)
            x_des_sx = phase_start_pos_sx + s * (0.0 - phase_start_pos_sx)

            model.dof_frictionloss[door_sx_dofadr] = FRICTION_MOVING
            model.dof_frictionloss[door_dx_dofadr] = FRICTION_CLOSED

            error_sx = x_des_sx - door_sx_pos

            force_sx = ACT_TO_DOOR_SIGN_SX * (K_CLOSE * error_sx - D_CLOSE * door_sx_vel)
            motor_ctrl_sx = np.clip(force_sx, -FORCE_MAX, FORCE_MAX)

            # motor_ctrl_dx = hold_closed_force(ACT_TO_DOOR_SIGN_DX, door_dx_pos, door_dx_vel)

            if phase_timer >= T_CLOSE:
                phase = WAIT_CLOSED_SX
                phase_start_time = data.time
                phase_start_pos_sx = data.qpos[door_sx_qposadr]
                print("Phase: WAIT_CLOSED_SX")

        # PHASE 3: WAIT WITH LEFT DOOR CLOSED
        elif phase == WAIT_CLOSED_SX:

            model.dof_frictionloss[door_sx_dofadr] = FRICTION_CLOSED
            model.dof_frictionloss[door_dx_dofadr] = FRICTION_CLOSED

            motor_ctrl_sx = hold_closed_force(ACT_TO_DOOR_SIGN_SX, door_sx_pos, door_sx_vel)
            # motor_ctrl_dx = hold_closed_force(ACT_TO_DOOR_SIGN_DX, door_dx_pos, door_dx_vel)

            if phase_timer >= T_WAIT_CLOSED:
                phase = OPEN_DX
                phase_start_time = data.time
                phase_start_pos_dx = data.qpos[door_dx_qposadr]
                print("Phase: OPEN_DX")

        # PHASE 4: OPEN RIGHT DOOR
        elif phase == OPEN_DX:

            s = smoothstep(phase_timer / T_OPEN)
            x_des_dx = phase_start_pos_dx + s * (OPEN_DISTANCE - phase_start_pos_dx)

            model.dof_frictionloss[door_sx_dofadr] = FRICTION_CLOSED
            model.dof_frictionloss[door_dx_dofadr] = FRICTION_CLOSED if door_dx_pos < 0.01 else FRICTION_MOVING

            # motor_ctrl_sx = hold_closed_force(ACT_TO_DOOR_SIGN_SX, door_sx_pos, door_sx_vel)

            error_dx = x_des_dx - door_dx_pos

            force_dx = ACT_TO_DOOR_SIGN_DX * (K_OPEN * error_dx - D_OPEN * door_dx_vel)
            motor_ctrl_dx = np.clip(force_dx, -FORCE_MAX, FORCE_MAX)

            if phase_timer >= T_OPEN:
                phase = WAIT_OPEN_DX
                phase_start_time = data.time
                phase_start_pos_dx = data.qpos[door_dx_qposadr]
                print("Phase: WAIT_OPEN_DX")

        # PHASE 5: WAIT WITH RIGHT DOOR OPEN
        elif phase == WAIT_OPEN_DX:

            # motor_ctrl_sx = hold_closed_force(ACT_TO_DOOR_SIGN_SX, door_sx_pos, door_sx_vel)

            error_dx = OPEN_DISTANCE - door_dx_pos

            force_dx = ACT_TO_DOOR_SIGN_DX * (K_HOLD * error_dx - D_HOLD * door_dx_vel)
            motor_ctrl_dx = np.clip(force_dx, -FORCE_MAX, FORCE_MAX)

            if phase_timer >= T_WAIT_OPEN:
                phase = CLOSE_DX
                phase_start_time = data.time
                phase_start_pos_dx = data.qpos[door_dx_qposadr]
                print("Phase: CLOSE_DX")

        # PHASE 6: CLOSE RIGHT DOOR
        elif phase == CLOSE_DX:

            s = smoothstep(phase_timer / T_CLOSE)
            x_des_dx = phase_start_pos_dx + s * (0.0 - phase_start_pos_dx)

            model.dof_frictionloss[door_sx_dofadr] = FRICTION_CLOSED
            model.dof_frictionloss[door_dx_dofadr] = FRICTION_MOVING

            # motor_ctrl_sx = hold_closed_force(ACT_TO_DOOR_SIGN_SX, door_sx_pos, door_sx_vel)

            error_dx = x_des_dx - door_dx_pos

            force_dx = ACT_TO_DOOR_SIGN_DX * (K_CLOSE * error_dx - D_CLOSE * door_dx_vel)
            motor_ctrl_dx = np.clip(force_dx, -FORCE_MAX, FORCE_MAX)

            if phase_timer >= T_CLOSE:
                phase = WAIT_CLOSED_DX
                phase_start_time = data.time
                phase_start_pos_dx = data.qpos[door_dx_qposadr]
                print("Phase: WAIT_CLOSED_DX")

        # PHASE 7: WAIT WITH RIGHT DOOR CLOSED
        elif phase == WAIT_CLOSED_DX:

            model.dof_frictionloss[door_sx_dofadr] = FRICTION_CLOSED
            model.dof_frictionloss[door_dx_dofadr] = FRICTION_CLOSED

            # motor_ctrl_sx = hold_closed_force(ACT_TO_DOOR_SIGN_SX, door_sx_pos, door_sx_vel)
            motor_ctrl_dx = hold_closed_force(ACT_TO_DOOR_SIGN_DX, door_dx_pos, door_dx_vel)

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

        # ACTUATOR FORCE AND POWER
        actuator_force_sx = data.qfrc_actuator[act_sx_dofadr]
        actuator_force_dx = data.qfrc_actuator[act_dx_dofadr]

        actuator_vel_sx = data.qvel[act_sx_dofadr]
        actuator_vel_dx = data.qvel[act_dx_dofadr]

        power_sx = actuator_force_sx * actuator_vel_sx
        power_dx = actuator_force_dx * actuator_vel_dx

        # LOGGING
        log_time.append(data.time)

        log_door_sx_m.append(data.qpos[door_sx_qposadr])
        log_door_sx_vel_m_s.append(data.qvel[door_sx_dofadr])
        log_act_sx_m.append(data.qpos[act_sx_qposadr])
        log_act_sx_vel_m_s.append(data.qvel[act_sx_dofadr])
        log_align_sx_deg.append(quat_angle_deg(data.qpos[align_sx_qposadr:align_sx_qposadr + 4]))

        log_door_dx_m.append(data.qpos[door_dx_qposadr])
        log_door_dx_vel_m_s.append(data.qvel[door_dx_dofadr])
        log_act_dx_m.append(data.qpos[act_dx_qposadr])
        log_act_dx_vel_m_s.append(data.qvel[act_dx_dofadr])
        log_align_dx_deg.append(quat_angle_deg(data.qpos[align_dx_qposadr:align_dx_qposadr + 4]))

        log_force_sx.append(actuator_force_sx)
        log_force_dx.append(actuator_force_dx)

        log_power_sx.append(power_sx)
        log_power_dx.append(power_dx)

        # PRINT DIAGNOSTICS
        if len(log_time) % 50 == 0:
            print(
                f"Time {data.time:.2f} | phase {phase} | "
                f"porta_sx: {data.qpos[door_sx_qposadr] * 1000:.1f} mm | "
                f"attuatore_sx: {data.qpos[act_sx_qposadr] * 1000:.1f} mm | "
                f"align_sx: {log_align_sx_deg[-1]:.2f} deg | "
                f"porta_dx: {data.qpos[door_dx_qposadr] * 1000:.1f} mm | "
                f"attuatore_dx: {data.qpos[act_dx_qposadr] * 1000:.1f} mm | "
                f"align_dx: {log_align_dx_deg[-1]:.2f} deg"
            )


# FINAL OUTPUT
door_sx_pos_mm = data.qpos[door_sx_qposadr] * 1000.0
door_dx_pos_mm = data.qpos[door_dx_qposadr] * 1000.0

print(f"Final door SX position: {door_sx_pos_mm:.2f} mm")
print(f"Final door DX position: {door_dx_pos_mm:.2f} mm")

scipy.io.savemat("dati_porta_lineare_osaka.mat", {
    "tempo": np.array(log_time),

    "posizione_porta_sx_m": np.array(log_door_sx_m),
    "velocita_porta_sx_m_s": np.array(log_door_sx_vel_m_s),
    "posizione_attuatore_lineare_sx_m": np.array(log_act_sx_m),
    "velocita_attuatore_lineare_sx_m_s": np.array(log_act_sx_vel_m_s),
    "rotazione_allineamento_sx_deg": np.array(log_align_sx_deg),
    "forza_attuatore_sx": np.array(log_force_sx),
    "potenza_meccanica_sx": np.array(log_power_sx),

    "posizione_porta_dx_m": np.array(log_door_dx_m),
    "velocita_porta_dx_m_s": np.array(log_door_dx_vel_m_s),
    "posizione_attuatore_lineare_dx_m": np.array(log_act_dx_m),
    "velocita_attuatore_lineare_dx_m_s": np.array(log_act_dx_vel_m_s),
    "rotazione_allineamento_dx_deg": np.array(log_align_dx_deg),
    "forza_attuatore_dx": np.array(log_force_dx),
    "potenza_meccanica_dx": np.array(log_power_dx),
})

print("Dati salvati in: dati_porta_lineare_osaka.mat")