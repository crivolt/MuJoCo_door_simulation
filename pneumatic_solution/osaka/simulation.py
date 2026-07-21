import mujoco
import mujoco.viewer
import numpy as np
import time
import scipy.io


# LOAD MODEL
model = mujoco.MjModel.from_xml_path("/Users/cristianvoltan/Desktop/unipd/tirocinio/CAD/pneumatic_solution/osaka/osaka_pneumatic.xml")
data = mujoco.MjData(model)


# FUNCTION TO GET MODEL IDS
def get_id(obj_type, name):
    obj_id = mujoco.mj_name2id(model, obj_type, name)
    return obj_id


# DOOR JOINTS
door_sx_joint_id = get_id(mujoco.mjtObj.mjOBJ_JOINT, "slide_sx")
door_sx_qposadr = model.jnt_qposadr[door_sx_joint_id]
door_sx_dofadr = model.jnt_dofadr[door_sx_joint_id]

door_dx_joint_id = get_id(mujoco.mjtObj.mjOBJ_JOINT, "slide_dx")
door_dx_qposadr = model.jnt_qposadr[door_dx_joint_id]
door_dx_dofadr = model.jnt_dofadr[door_dx_joint_id]


# CYLINDER JOINTS
cylinder_sx_joint_id = get_id(mujoco.mjtObj.mjOBJ_JOINT, "corsa_cilindro_sx")
cylinder_sx_qposadr = model.jnt_qposadr[cylinder_sx_joint_id]

cylinder_dx_joint_id = get_id(mujoco.mjtObj.mjOBJ_JOINT, "corsa_cilindro_dx")
cylinder_dx_qposadr = model.jnt_qposadr[cylinder_dx_joint_id]


# BALL JOINTS BETWEEN STEM AND BRACKET
snodo_sx_joint_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "snodo_stelo_staffa_sx")
snodo_dx_joint_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "snodo_stelo_staffa_dx")

snodo_sx_qposadr = model.jnt_qposadr[snodo_sx_joint_id] if snodo_sx_joint_id != -1 else None
snodo_dx_qposadr = model.jnt_qposadr[snodo_dx_joint_id] if snodo_dx_joint_id != -1 else None


# ACTUATORS
actuator_sx_id = get_id(mujoco.mjtObj.mjOBJ_ACTUATOR, "forza_pneumatica_sx")
actuator_dx_id = get_id(mujoco.mjtObj.mjOBJ_ACTUATOR, "forza_pneumatica_dx")


# EQUALITY CONSTRAINTS BETWEEN BRACKET AND HANDLE
connect_sx_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_EQUALITY, "connessione_staffa_maniglia_sx")
connect_dx_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_EQUALITY, "connessione_staffa_maniglia_dx")

if connect_sx_id != -1:
    data.eq_active[connect_sx_id] = 1

if connect_dx_id != -1:
    data.eq_active[connect_dx_id] = 1


# INITIAL FORWARD
mujoco.mj_forward(model, data)


# SIMULATION PARAMETERS
NUM_CYCLES = 1
current_cycle = 0

OPEN_POS_SX = 0.70
OPEN_POS_DX = 0.70

# Timeout di sicurezza per ogni fase di movimento (evita loop infiniti se il
# sistema non converge mai, es. per forza insufficiente o guadagni sbagliati)
T_OPEN_SX_TIMEOUT = 1.0
T_CLOSE_SX_TIMEOUT = 1.0

T_OPEN_DX_TIMEOUT = 1.0
T_CLOSE_DX_TIMEOUT = 1.0

T_WAIT_OPEN_TO_CLOSE_SX = 4.0
T_WAIT_BETWEEN_APERTURES = 3.0
T_WAIT_OPEN_TO_CLOSE_DX = 4.0
T_WAIT_CLOSED_END = 5.0

F_MAX_SX = 120.0
F_MAX_DX = 120.0

# Guadagni PD: D vicino al valore di smorzamento critico rispetto a K
# (aumentare D se il movimento oscilla/rimbalza, aumentare K se troppo lento)
K_OPEN_SX = 2500.0
D_OPEN_SX = 150.0

K_OPEN_DX = 2500.0
D_OPEN_DX = 150.0

K_CLOSE_SX = 2500.0
D_CLOSE_SX = 150.0

K_CLOSE_DX = 2500.0
D_CLOSE_DX = 150.0

K_HOLD_SX = 2000.0
D_HOLD_SX = 250.0

K_HOLD_DX = 2000.0
D_HOLD_DX = 250.0

FRICTION_CLOSED_SX = 33.0
FRICTION_MOVING_SX = 3.0

FRICTION_CLOSED_DX = 33.0
FRICTION_MOVING_DX = 3.0

# Soglie di convergenza per considerare "arrivata" la porta a fine fase
POS_TOL = 0.002   # tolleranza sulla posizione [m]
VEL_TOL = 0.005   # tolleranza sulla velocita' [m/s]


# STATE MACHINE
OPEN_SX = 0
WAIT_SX_OPEN = 1
CLOSE_SX = 2
WAIT_BETWEEN = 3
OPEN_DX = 4
WAIT_DX_OPEN = 5
CLOSE_DX = 6
WAIT_CLOSED_END = 7

phase = OPEN_SX
phase_start_time = 0.0
phase_start_pos_sx = data.qpos[door_sx_qposadr]
phase_start_pos_dx = data.qpos[door_dx_qposadr]


# LOGGING
log_time = []

log_door_sx_pos = []
log_door_dx_pos = []

log_door_sx_vel = []
log_door_dx_vel = []

log_cylinder_sx_pos = []
log_cylinder_dx_pos = []

log_force_sx = []
log_force_dx = []

log_snodo_sx_quat = []
log_snodo_dx_quat = []


# TEST ACTUATOR DIRECTION
def test_open_direction(actuator_id, door_qposadr, force_value):
    test_data = mujoco.MjData(model)

    if connect_sx_id != -1:
        test_data.eq_active[connect_sx_id] = 1

    if connect_dx_id != -1:
        test_data.eq_active[connect_dx_id] = 1

    mujoco.mj_forward(model, test_data)

    for _ in range(500):
        test_data.ctrl[actuator_id] = force_value
        mujoco.mj_step(model, test_data)

    return test_data.qpos[door_qposadr]


sx_positive = test_open_direction(actuator_sx_id, door_sx_qposadr, 300.0)
sx_negative = test_open_direction(actuator_sx_id, door_sx_qposadr, -300.0)

dx_positive = test_open_direction(actuator_dx_id, door_dx_qposadr, 300.0)
dx_negative = test_open_direction(actuator_dx_id, door_dx_qposadr, -300.0)

OPEN_SIGN_SX = 1.0 if sx_positive > sx_negative else -1.0
OPEN_SIGN_DX = 1.0 if dx_positive > dx_negative else -1.0

print(f"Opening sign SX selected: {OPEN_SIGN_SX:+.0f}")
print(f"SX position with +300 N: {sx_positive:.3f} m")
print(f"SX position with -300 N: {sx_negative:.3f} m")

print(f"Opening sign DX selected: {OPEN_SIGN_DX:+.0f}")
print(f"DX position with +300 N: {dx_positive:.3f} m")
print(f"DX position with -300 N: {dx_negative:.3f} m")


# RESET AFTER DIRECTION TEST
data = mujoco.MjData(model)

if connect_sx_id != -1:
    data.eq_active[connect_sx_id] = 1

if connect_dx_id != -1:
    data.eq_active[connect_dx_id] = 1

mujoco.mj_forward(model, data)


# CONTROL FUNCTION
def pd_force(pos_des, pos, vel, open_sign, K, D, F_MAX):
    force_cmd = open_sign * (K * (pos_des - pos) - D * vel)
    return np.clip(force_cmd, -F_MAX, F_MAX)


# CONVERGENCE CHECK
def converged(pos_des, pos, vel):
    return abs(pos_des - pos) < POS_TOL and abs(vel) < VEL_TOL


# PHASE CHANGE FUNCTION
def set_phase(new_phase):
    global phase, phase_start_time, phase_start_pos_sx, phase_start_pos_dx

    phase = new_phase
    phase_start_time = data.time
    phase_start_pos_sx = data.qpos[door_sx_qposadr]
    phase_start_pos_dx = data.qpos[door_dx_qposadr]


set_phase(OPEN_SX)


# SIMULATION LOOP
with mujoco.viewer.launch_passive(model, data) as viewer:

    while viewer.is_running() and current_cycle < NUM_CYCLES:

        phase_timer = data.time - phase_start_time

        door_sx_pos = data.qpos[door_sx_qposadr]
        door_dx_pos = data.qpos[door_dx_qposadr]

        door_sx_vel = data.qvel[door_sx_dofadr]
        door_dx_vel = data.qvel[door_dx_dofadr]

        force_sx = 0.0
        force_dx = 0.0


        # OPEN LEFT DOOR
        if phase == OPEN_SX:

            pos_des_sx = OPEN_POS_SX
            pos_des_dx = 0.0

            model.dof_frictionloss[door_sx_dofadr] = FRICTION_CLOSED_SX if door_sx_pos < 0.01 else FRICTION_MOVING_SX
            model.dof_frictionloss[door_dx_dofadr] = FRICTION_CLOSED_DX

            force_sx = pd_force(pos_des_sx, door_sx_pos, door_sx_vel, OPEN_SIGN_SX, K_OPEN_SX, D_OPEN_SX, F_MAX_SX)
            force_dx = pd_force(pos_des_dx, door_dx_pos, door_dx_vel, OPEN_SIGN_DX, K_HOLD_DX, D_HOLD_DX, F_MAX_DX)

            done = converged(pos_des_sx, door_sx_pos, door_sx_vel)
            if done or phase_timer >= T_OPEN_SX_TIMEOUT:
                set_phase(WAIT_SX_OPEN)
                print("Phase: WAIT_SX_OPEN")


        # WAIT WITH LEFT DOOR OPEN
        elif phase == WAIT_SX_OPEN:

            pos_des_sx = OPEN_POS_SX
            pos_des_dx = 0.0

            model.dof_frictionloss[door_sx_dofadr] = FRICTION_MOVING_SX
            model.dof_frictionloss[door_dx_dofadr] = FRICTION_CLOSED_DX

            force_sx = pd_force(pos_des_sx, door_sx_pos, door_sx_vel, OPEN_SIGN_SX, K_HOLD_SX, D_HOLD_SX, F_MAX_SX)
            force_dx = pd_force(pos_des_dx, door_dx_pos, door_dx_vel, OPEN_SIGN_DX, K_HOLD_DX, D_HOLD_DX, F_MAX_DX)

            if phase_timer >= T_WAIT_OPEN_TO_CLOSE_SX:
                set_phase(CLOSE_SX)
                print("Phase: CLOSE_SX")


        # CLOSE LEFT DOOR
        elif phase == CLOSE_SX:

            pos_des_sx = 0.0
            pos_des_dx = 0.0

            model.dof_frictionloss[door_sx_dofadr] = FRICTION_MOVING_SX
            model.dof_frictionloss[door_dx_dofadr] = FRICTION_CLOSED_DX

            force_sx = pd_force(pos_des_sx, door_sx_pos, door_sx_vel, OPEN_SIGN_SX, K_CLOSE_SX, D_CLOSE_SX, F_MAX_SX)
            force_dx = pd_force(pos_des_dx, door_dx_pos, door_dx_vel, OPEN_SIGN_DX, K_HOLD_DX, D_HOLD_DX, F_MAX_DX)

            done = converged(pos_des_sx, door_sx_pos, door_sx_vel)
            if done or phase_timer >= T_CLOSE_SX_TIMEOUT:
                set_phase(WAIT_BETWEEN)
                print("Phase: WAIT_BETWEEN")


        # WAIT BETWEEN LEFT AND RIGHT OPENINGS
        elif phase == WAIT_BETWEEN:

            pos_des_sx = 0.0
            pos_des_dx = 0.0

            model.dof_frictionloss[door_sx_dofadr] = FRICTION_CLOSED_SX
            model.dof_frictionloss[door_dx_dofadr] = FRICTION_CLOSED_DX

            force_sx = pd_force(pos_des_sx, door_sx_pos, door_sx_vel, OPEN_SIGN_SX, K_HOLD_SX, D_HOLD_SX, F_MAX_SX)
            force_dx = pd_force(pos_des_dx, door_dx_pos, door_dx_vel, OPEN_SIGN_DX, K_HOLD_DX, D_HOLD_DX, F_MAX_DX)

            if phase_timer >= T_WAIT_BETWEEN_APERTURES:
                set_phase(OPEN_DX)
                print("Phase: OPEN_DX")


        # OPEN RIGHT DOOR
        elif phase == OPEN_DX:

            pos_des_sx = 0.0
            pos_des_dx = OPEN_POS_DX

            model.dof_frictionloss[door_sx_dofadr] = FRICTION_CLOSED_SX
            model.dof_frictionloss[door_dx_dofadr] = FRICTION_CLOSED_DX if door_dx_pos < 0.01 else FRICTION_MOVING_DX

            force_sx = pd_force(pos_des_sx, door_sx_pos, door_sx_vel, OPEN_SIGN_SX, K_HOLD_SX, D_HOLD_SX, F_MAX_SX)
            force_dx = pd_force(pos_des_dx, door_dx_pos, door_dx_vel, OPEN_SIGN_DX, K_OPEN_DX, D_OPEN_DX, F_MAX_DX)

            done = converged(pos_des_dx, door_dx_pos, door_dx_vel)
            if done or phase_timer >= T_OPEN_DX_TIMEOUT:
                set_phase(WAIT_DX_OPEN)
                print("Phase: WAIT_DX_OPEN")


        # WAIT WITH RIGHT DOOR OPEN
        elif phase == WAIT_DX_OPEN:

            pos_des_sx = 0.0
            pos_des_dx = OPEN_POS_DX

            model.dof_frictionloss[door_sx_dofadr] = FRICTION_CLOSED_SX
            model.dof_frictionloss[door_dx_dofadr] = FRICTION_MOVING_DX

            force_sx = pd_force(pos_des_sx, door_sx_pos, door_sx_vel, OPEN_SIGN_SX, K_HOLD_SX, D_HOLD_SX, F_MAX_SX)
            force_dx = pd_force(pos_des_dx, door_dx_pos, door_dx_vel, OPEN_SIGN_DX, K_HOLD_DX, D_HOLD_DX, F_MAX_DX)

            if phase_timer >= T_WAIT_OPEN_TO_CLOSE_DX:
                set_phase(CLOSE_DX)
                print("Phase: CLOSE_DX")


        # CLOSE RIGHT DOOR
        elif phase == CLOSE_DX:

            pos_des_sx = 0.0
            pos_des_dx = 0.0

            model.dof_frictionloss[door_sx_dofadr] = FRICTION_CLOSED_SX
            model.dof_frictionloss[door_dx_dofadr] = FRICTION_MOVING_DX

            force_sx = pd_force(pos_des_sx, door_sx_pos, door_sx_vel, OPEN_SIGN_SX, K_HOLD_SX, D_HOLD_SX, F_MAX_SX)
            force_dx = pd_force(pos_des_dx, door_dx_pos, door_dx_vel, OPEN_SIGN_DX, K_CLOSE_DX, D_CLOSE_DX, F_MAX_DX)

            done = converged(pos_des_dx, door_dx_pos, door_dx_vel)
            if done or phase_timer >= T_CLOSE_DX_TIMEOUT:
                set_phase(WAIT_CLOSED_END)
                print("Phase: WAIT_CLOSED_END")


        # WAIT WITH BOTH DOORS CLOSED
        elif phase == WAIT_CLOSED_END:

            pos_des_sx = 0.0
            pos_des_dx = 0.0

            model.dof_frictionloss[door_sx_dofadr] = FRICTION_CLOSED_SX
            model.dof_frictionloss[door_dx_dofadr] = FRICTION_CLOSED_DX

            force_sx = pd_force(pos_des_sx, door_sx_pos, door_sx_vel, OPEN_SIGN_SX, K_HOLD_SX, D_HOLD_SX, F_MAX_SX)
            force_dx = pd_force(pos_des_dx, door_dx_pos, door_dx_vel, OPEN_SIGN_DX, K_HOLD_DX, D_HOLD_DX, F_MAX_DX)

            if phase_timer >= T_WAIT_CLOSED_END:
                current_cycle += 1
                print(f"=== Cycle {current_cycle}/{NUM_CYCLES} completed ===")

                if current_cycle >= NUM_CYCLES:
                    break

                set_phase(OPEN_SX)
                print("Phase: OPEN_SX")


        # SEND CONTROL
        data.ctrl[actuator_sx_id] = force_sx
        data.ctrl[actuator_dx_id] = force_dx


        # STEP SIMULATION
        mujoco.mj_step(model, data)
        viewer.sync()
        time.sleep(model.opt.timestep)


        # LOGGING
        log_time.append(data.time)

        log_door_sx_pos.append(data.qpos[door_sx_qposadr])
        log_door_dx_pos.append(data.qpos[door_dx_qposadr])

        log_door_sx_vel.append(data.qvel[door_sx_dofadr])
        log_door_dx_vel.append(data.qvel[door_dx_dofadr])

        log_cylinder_sx_pos.append(data.qpos[cylinder_sx_qposadr])
        log_cylinder_dx_pos.append(data.qpos[cylinder_dx_qposadr])

        log_force_sx.append(force_sx)
        log_force_dx.append(force_dx)

        if snodo_sx_qposadr is not None:
            log_snodo_sx_quat.append(data.qpos[snodo_sx_qposadr:snodo_sx_qposadr + 4].copy())

        if snodo_dx_qposadr is not None:
            log_snodo_dx_quat.append(data.qpos[snodo_dx_qposadr:snodo_dx_qposadr + 4].copy())


        # PRINT DIAGNOSTICS
        if len(log_time) % 50 == 0:
            print(
                f"Time {data.time:.2f} | phase {phase} | "
                f"SX: {data.qpos[door_sx_qposadr]:.3f} m | "
                f"DX: {data.qpos[door_dx_qposadr]:.3f} m | "
                f"cyl SX: {data.qpos[cylinder_sx_qposadr]:.3f} m | "
                f"cyl DX: {data.qpos[cylinder_dx_qposadr]:.3f} m | "
                f"F SX: {force_sx:.2f} N | "
                f"F DX: {force_dx:.2f} N"
            )


# FINAL OUTPUT
door_sx_pos_final = data.qpos[door_sx_qposadr]
door_dx_pos_final = data.qpos[door_dx_qposadr]

print(f"Final SX door position: {door_sx_pos_final:.3f} m")
print(f"Final DX door position: {door_dx_pos_final:.3f} m")


# SAVE DATA
mat_data = {
    "tempo": np.array(log_time),

    "posizione_porta_sx": np.array(log_door_sx_pos),
    "posizione_porta_dx": np.array(log_door_dx_pos),

    "velocita_porta_sx": np.array(log_door_sx_vel),
    "velocita_porta_dx": np.array(log_door_dx_vel),

    "posizione_cilindro_sx": np.array(log_cylinder_sx_pos),
    "posizione_cilindro_dx": np.array(log_cylinder_dx_pos),

    "forza_sx": np.array(log_force_sx),
    "forza_dx": np.array(log_force_dx)
}

if len(log_snodo_sx_quat) > 0:
    mat_data["quat_snodo_stelo_staffa_sx"] = np.array(log_snodo_sx_quat)

if len(log_snodo_dx_quat) > 0:
    mat_data["quat_snodo_stelo_staffa_dx"] = np.array(log_snodo_dx_quat)

scipy.io.savemat("dati_porta_pneumatica_osaka.mat", mat_data)