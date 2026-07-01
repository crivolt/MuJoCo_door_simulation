import mujoco
import mujoco.viewer
import numpy as np
import scipy.io


#  PARAMETERS
N_CYCLES = 1
OPEN_DEG = 62  

XML_PATH      = "/Users/cristianvoltan/Desktop/unipd/tirocinio/CAD/Rail/osaka_3p/Osaka_3p.xml"
TRAJ_SX_PATH  = "/Users/cristianvoltan/Desktop/unipd/tirocinio/CAD/Rail/osaka_3p/traiettorie/traiettoria_sinistra.npy"
TRAJ_DX_PATH  = "/Users/cristianvoltan/Desktop/unipd/tirocinio/CAD/Rail/osaka_3p/traiettorie/traiettoria_destra.npy"


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
# biasprm[2] = -Dq
K_APPROACH = 700.0
D_APPROACH = 200.0

K_GRASP    = 800.0
D_GRASP    = 120.0

K_OPENING  = 2500.0
D_OPENING  = 180.0

K_CLOSING  = 2000.0
D_CLOSING  = 150.0

K_HOLD     = 2500.0
D_HOLD     = 300.0

JOINT_FORCE_LIMITS = np.array([330.0, 330.0, 150.0, 56.0, 56.0, 56.0])

SLITTA_KP = 3000.0
SLITTA_KD = 2500.0


# LOAD MODEL

model = mujoco.MjModel.from_xml_path(XML_PATH)
data  = mujoco.MjData(model)

ACT_SLITTA = 0
ACT_ARM    = [1, 2, 3, 4, 5, 6]  # shoulder_pan … wrist_3


def set_joint_impedance(Kq, Dq, force_limits):
    for i, a in enumerate(ACT_ARM):
        model.actuator_gainprm[a, 0] =  Kq
        model.actuator_biasprm[a, 1] = -Kq
        model.actuator_biasprm[a, 2] = -Dq
        model.actuator_forcerange[a, 0] = -force_limits[i]
        model.actuator_forcerange[a, 1] =  force_limits[i]


set_joint_impedance(K_APPROACH, D_APPROACH, JOINT_FORCE_LIMITS)


# JOINTS
all_joint_names = ["shoulder_pan_joint", "shoulder_lift_joint", "elbow_joint", "wrist_1_joint", "wrist_2_joint", "wrist_3_joint"]

all_qposadr, all_dofadr = [], []
for name in all_joint_names:
    jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)
    all_qposadr.append(model.jnt_qposadr[jid])
    all_dofadr.append(model.jnt_dofadr[jid])

all_qposadr = np.array(all_qposadr)
all_dofadr  = np.array(all_dofadr)


# CHASSIS
slitta_jid     = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "giunto8_slitta")
slitta_qposadr = model.jnt_qposadr[slitta_jid]
slitta_dofadr  = model.jnt_dofadr[slitta_jid]

SLITTA_NEUTRAL = 0.0
SLITTA_POS_SX  = -0.82
SLITTA_POS_DX  =  0.83

# NEUTRAL POSITION
Q_NEUTRAL = np.array([-1.57, -1.57, -1.57, 0.0, 0.0, 0.0])

FRICTION_STATIC  = 5.28
FRICTION_DYNAMIC = 1.056

# CUBIC POLYNOMIAL PROFILE
# θ(s) = a0 + a1·s + a2·s² + a3·s³,  s ∈ [0, 1]
#
# Boundary conditions:
#   θ(0) = 0        → starts at zero
#   θ(1) = theta_f  → ends at target
#   ω(0) = 0        → zero velocity at start
#   ω(1) = 0        → zero velocity at end

def compute_cubic(theta_f):
    """Solve for cubic polynomial coefficients [a0, a1, a2, a3]."""
    A = np.array([
        [1, 0, 0, 0],
        [1, 1, 1, 1],
        [0, 1, 0, 0],
        [0, 1, 2, 3],
    ])
    b = np.array([0, theta_f, 0, 0])
    return np.linalg.solve(A, b)


def cubic_theta(s, c):
    """Evaluate cubic polynomial at normalized time s."""
    s = np.clip(s, 0.0, 1.0)
    return c[0] + c[1]*s + c[2]*s**2 + c[3]*s**3


# Cubic coefficients (normalized 0→1 for trajectory index mapping)
cubic_c       = compute_cubic(theta_f=OPEN_DEG)  
T_OPEN        = 1.0   
T_CLOSE       = 1.0   

open_start_time  = None
close_start_time = None


# TRAJECTORIES
def build_traj_close(traj, open_deg, max_deg=95):
    n = max(2, min(int(len(traj) * open_deg / max_deg), len(traj)))
    return traj[:n][::-1].copy(), n

trajectory_sx = np.load(TRAJ_SX_PATH)
trajectory_dx = np.load(TRAJ_DX_PATH)

traj_close_sx, n_open_sx = build_traj_close(trajectory_sx, OPEN_DEG)
traj_close_dx, n_open_dx = build_traj_close(trajectory_dx, OPEN_DEG)


#  DOOR CONFIGURATIONS
door_configs = {
    "sx": dict(
        label         = "SINISTRA",
        slitta_target = SLITTA_POS_SX,
        q_home        = np.array([-1.51, -1.51, 1.51, -1.51, 0.0, 0.0]),
        site_name     = "attachment_site",
        target_site   = "target_manigliasx",
        weld_name     = "presa_sx",
        door_joint    = "giunto_porta_sx",
        trajectory    = trajectory_sx,
        traj_close    = traj_close_sx,
        n_open        = n_open_sx,
    ),
    "dx": dict(
        label         = "DESTRA",
        slitta_target = SLITTA_POS_DX,
        q_home        = np.array([-1.63, -1.63, -1.48, -1.51, 0.0, 0.0]),
        site_name     = "attachment_site",
        target_site   = "target_manigliadx",
        weld_name     = "presa_dx",
        door_joint    = "giunto_porta_dx",
        trajectory    = trajectory_dx,
        traj_close    = traj_close_dx,
        n_open        = n_open_dx,
    ),
}

DOOR_SEQUENCE = ["sx", "dx"]

#  STATE MACHINE
MOVE_TO_DOOR    = 0
APPROACH_HANDLE = 1
CLOSE_GRIPPER   = 2
STABILIZE_WELD  = 3
OPEN_DOOR       = 4
WAIT_OPEN       = 5
CLOSE_DOOR      = 6
HOLD_CLOSED     = 7
RETURN_NEUTRAL  = 8
DONE            = 9

STATE_NAMES = {
    MOVE_TO_DOOR:    "MOVE_TO_DOOR",
    APPROACH_HANDLE: "APPROACH",
    CLOSE_GRIPPER:   "CLOSE_GRIP",
    STABILIZE_WELD:  "STABILIZE",
    OPEN_DOOR:       "OPEN_DOOR",
    WAIT_OPEN:       "WAIT_OPEN",
    CLOSE_DOOR:      "CLOSE_DOOR",
    HOLD_CLOSED:     "HOLD",
    RETURN_NEUTRAL:  "RETURN",
    DONE:            "DONE",
}


#  PARAMETERS IK
KP_IK         = 1.20
KI_IK         = 0.10
INTEGRAL_CLIP = 0.05
damping       = 5e-3
ik_every      = 1

gain_phase0 = 0.003
gain_phase3 = 0.04
gain_close  = 0.03

CLOSE_DURATION          = 500
STALL_LIMIT             = 2000
WAIT_BEFORE_CLOSE_STEPS = int(13.0 / model.opt.timestep)


#  VARIABLES

error_integral = np.zeros(3)
q_des_all      = Q_NEUTRAL.copy()

site_id        = None
target_site_id = None
weld_id        = None
door_qposadr   = None
door_dofadr    = None
active         = {}

SLITTA_LOCKED   = False
SLITTA_LOCK_POS = SLITTA_NEUTRAL

q_interp_start      = Q_NEUTRAL.copy()
slitta_interp_start = SLITTA_NEUTRAL


#  HELPER

def reset_integral():
    global error_integral
    error_integral[:] = 0.0


def setup_door(key):
    global active, site_id, target_site_id, weld_id, door_qposadr, door_dofadr

    cfg = door_configs[key]
    active = cfg

    site_id        = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE,    cfg["site_name"])
    target_site_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE,    cfg["target_site"])
    weld_id        = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_EQUALITY, cfg["weld_name"])
    door_jid       = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT,   cfg["door_joint"])

    door_qposadr = model.jnt_qposadr[door_jid]
    door_dofadr  = model.jnt_dofadr[door_jid]
    q_des_all[:] = Q_NEUTRAL.copy()
    reset_integral()


def interpolate_arm(q_start, q_end, slitta_start, slitta_target):
    slitta_pos   = data.qpos[slitta_qposadr]
    total_travel = abs(slitta_target - slitta_start)
    traveled     = abs(slitta_pos - slitta_start)
    alpha        = np.clip(traveled / total_travel, 0.0, 1.0) if total_travel > 1e-6 else 1.0
    return (1.0 - alpha) * q_start + alpha * q_end


def solve_position_ik(target, gain):
    global q_des_all, error_integral

    mujoco.mj_forward(model, data)

    site_pos  = data.site_xpos[site_id].copy()
    error_pos = target - site_pos

    error_integral += error_pos * model.opt.timestep
    error_integral  = np.clip(error_integral, -INTEGRAL_CLIP, INTEGRAL_CLIP)

    control = KP_IK * error_pos + KI_IK * error_integral

    jacp = np.zeros((3, model.nv))
    mujoco.mj_jacSite(model, data, jacp, None, site_id)
    J = jacp[:, all_dofadr]

    A  = J @ J.T + damping * np.eye(3)
    dq = J.T @ np.linalg.solve(A, control)

    q_des_all += gain * dq

    for i, name in enumerate(all_joint_names):
        jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)
        if model.jnt_limited[jid]:
            lo, hi = model.jnt_range[jid]
            q_des_all[i] = np.clip(q_des_all[i], lo, hi)

    return np.linalg.norm(error_pos)


#  INITIAL STATE
for i in range(len(all_joint_names)):
    data.qpos[all_qposadr[i]] = Q_NEUTRAL[i]

data.qpos[slitta_qposadr] = SLITTA_NEUTRAL
data.ctrl[ACT_SLITTA]     = SLITTA_NEUTRAL
data.ctrl[ACT_ARM]        = Q_NEUTRAL

mujoco.mj_forward(model, data)

q_des_all[:] = Q_NEUTRAL.copy()

#  INITIALIZATION
door_index  = 0
cycle_count = 0

setup_door(DOOR_SEQUENCE[door_index])

q_interp_start      = Q_NEUTRAL.copy()
slitta_interp_start = SLITTA_NEUTRAL

phase       = MOVE_TO_DOOR
phase_timer = 0
traj_index  = 1
close_index = 0
best_dist   = float("inf")
stall_counter = 0

log_time     = []
log_door_deg = []
log_door_vel = []
log_tau      = []
log_porta    = []
log_slitta   = []

MAX_STEPS = int(40000 * N_CYCLES * len(DOOR_SEQUENCE)) + 10000


# SIMULATION

with mujoco.viewer.launch_passive(model, data) as viewer:

    for step in range(MAX_STEPS):

        if phase == DONE:
            break

        mujoco.mj_forward(model, data)

        match phase:

            # FASE 0: MOVE TO DOOR
            # Translate chassis to the target position while interpolating
            # arm joints toward q_home. Locks slitta when in position.
            case 0:
                set_joint_impedance(K_APPROACH, D_APPROACH, JOINT_FORCE_LIMITS)

                slitta_tgt = active["slitta_target"]
                slitta_pos = data.qpos[slitta_qposadr]
                slitta_vel = data.qvel[slitta_dofadr]

                q_des_all[:] = interpolate_arm(q_interp_start, active["q_home"], slitta_interp_start, slitta_tgt)

                data.ctrl[ACT_SLITTA] = slitta_tgt

                if abs(slitta_pos - slitta_tgt) < 0.01 and abs(slitta_vel) < 0.01:
                    SLITTA_LOCKED   = True
                    SLITTA_LOCK_POS = float(data.qpos[slitta_qposadr])

                    data.qpos[slitta_qposadr] = SLITTA_LOCK_POS
                    data.qvel[slitta_dofadr]  = 0.0
                    data.ctrl[ACT_SLITTA]     = SLITTA_LOCK_POS

                    data.qvel[all_dofadr] = 0.0
                    data.qacc[:] = 0.0
                    mujoco.mj_forward(model, data)
                    reset_integral()

                    q_des_all[:] = active["q_home"].copy()

                    phase         = APPROACH_HANDLE
                    phase_timer   = 0
                    best_dist     = float("inf")
                    stall_counter = 0

            # FASE 1: APPROACH HANDLE
            # Move end-effector to the door handle using IK.
            # Exits when position error is small or IK stalls.
            case 1:
                set_joint_impedance(K_APPROACH, D_APPROACH, JOINT_FORCE_LIMITS)

                if step % ik_every == 0:
                    target_pos = data.site_xpos[target_site_id].copy()
                    dist = solve_position_ik(target_pos, gain_phase0)

                    if dist < best_dist - 0.001:
                        best_dist     = dist
                        stall_counter = 0
                    else:
                        stall_counter += 1

                    if dist < 0.0005 or stall_counter >= STALL_LIMIT:
                        print("q_home:       ", active["q_home"])
                        print("q_des finale: ", q_des_all)
                        print("differenza:   ", q_des_all - active["q_home"])
                        reset_integral()
                        phase       = CLOSE_GRIPPER
                        phase_timer = 0

            # FASE 2: CLOSE GRIPPER
            # Wait for the gripper to close, then activate the weld constraint.
            case 2:
                set_joint_impedance(K_GRASP, D_GRASP, JOINT_FORCE_LIMITS)

                phase_timer += 1
                if phase_timer >= CLOSE_DURATION:
                    data.qvel[:] = 0.0
                    data.qacc[:] = 0.0
                    mujoco.mj_forward(model, data)
                    err_weld = np.linalg.norm(data.site_xpos[site_id] - data.site_xpos[target_site_id])
                    print("Errore posizione prima del weld:", err_weld)
                    data.eq_active[weld_id] = 1
                    mujoco.mj_forward(model, data)
                    q_des_all[:] = data.qpos[all_qposadr].copy()
                    data.qvel[:] = 0.0
                    data.qacc[:] = 0.0
                    mujoco.mj_forward(model, data)
                    reset_integral()
                    phase       = STABILIZE_WELD
                    phase_timer = 0

            # FASE 3: STABILIZE WELD
            # Let the system settle after the weld is activated.
            case 3:
                set_joint_impedance(K_GRASP, D_GRASP, JOINT_FORCE_LIMITS)

                phase_timer += 1
                data.eq_active[weld_id] = 1

                if phase_timer >= 500:
                    data.qvel[:] = 0.0
                    data.qacc[:] = 0.0
                    mujoco.mj_forward(model, data)
                    traj_index       = 1
                    open_start_time  = None
                    reset_integral()
                    phase = OPEN_DOOR

            # FASE 4: OPEN DOOR
            # Follow the opening trajectory using a cubic velocity profile.
            # s ∈ [0,1] is mapped through cubic_c to a trajectory index.
            # Friction drops from static to dynamic after 1°.
            case 4:
                set_joint_impedance(K_OPENING, D_OPENING, JOINT_FORCE_LIMITS)
                data.eq_active[weld_id] = 1

                if open_start_time is None:
                    open_start_time = step * model.opt.timestep

                t_open = step * model.opt.timestep - open_start_time
                s = np.clip(t_open / T_OPEN, 0.0, 1.0)

                # Map s through cubic profile to trajectory index
                theta_norm = cubic_theta(s, cubic_c) / OPEN_DEG
                traj_index = int(theta_norm * (active["n_open"] - 1))
                traj_index = np.clip(traj_index, 0, active["n_open"] - 1)

                # Door friction: high until latch releases at ~1°
                door_angle_now = np.rad2deg(abs(data.qpos[door_qposadr]))
                if door_angle_now < 1.0:
                    model.dof_frictionloss[door_dofadr] = FRICTION_STATIC
                else:
                    model.dof_frictionloss[door_dofadr] = FRICTION_DYNAMIC

                solve_position_ik(active["trajectory"][traj_index, 0:3], gain_phase3)

                if s >= 1.0:
                    reset_integral()
                    q_des_all[:] = data.qpos[all_qposadr].copy()
                    open_start_time = None
                    phase       = WAIT_OPEN
                    phase_timer = 0

            # FASE 5: WAIT WITH DOOR OPEN
            # Hold position for a fixed time before closing.
            case 5:
                set_joint_impedance(K_HOLD, D_HOLD, JOINT_FORCE_LIMITS)

                data.eq_active[weld_id] = 1
                phase_timer += 1

                if phase_timer >= WAIT_BEFORE_CLOSE_STEPS:
                    print(f"[PRE-CLOSE] porta={active['label']}")
                    print(f"  q_des  = {np.round(q_des_all, 3)}")
                    print(f"  q_real = {np.round(data.qpos[all_qposadr], 3)}")
                    print(f"  diff   = {np.round(q_des_all - data.qpos[all_qposadr], 3)}")
                    print(f"  qvel   = {np.round(data.qvel[all_dofadr], 3)}")
                    print(f"  integral = {np.round(error_integral, 4)}")
                    data.qvel[:] = 0.0
                    data.qacc[:] = 0.0
                    mujoco.mj_forward(model, data)
                    q_des_all[:]     = data.qpos[all_qposadr].copy()
                    close_index      = 0
                    close_start_time = None
                    reset_integral()
                    phase = CLOSE_DOOR

            # FASE 6: CLOSE DOOR
            # Follow the closing trajectory (reverse of opening) using the
            # same cubic velocity profile.
            case 6:
                set_joint_impedance(K_CLOSING, D_CLOSING, JOINT_FORCE_LIMITS)
                data.eq_active[weld_id] = 1

                if close_start_time is None:
                    close_start_time = step * model.opt.timestep

                t_close = step * model.opt.timestep - close_start_time
                s = np.clip(t_close / T_CLOSE, 0.0, 1.0)

                # Map s through cubic profile to trajectory index
                theta_norm  = cubic_theta(s, cubic_c) / OPEN_DEG
                close_index = int(theta_norm * (len(active["traj_close"]) - 1))
                close_index = np.clip(close_index, 0, len(active["traj_close"]) - 1)

                solve_position_ik(active["traj_close"][close_index, 0:3], gain_close)

                if s >= 1.0:
                    reset_integral()
                    q_des_all[:] = data.qpos[all_qposadr].copy()
                    close_start_time = None
                    phase       = HOLD_CLOSED
                    phase_timer = 0

            # FASE 7: HOLD CLOSED
            # Hold the closed position, then release weld and return.
            case 7:
                set_joint_impedance(K_HOLD, D_HOLD, JOINT_FORCE_LIMITS)

                data.eq_active[weld_id] = 1
                phase_timer += 1

                if phase_timer >= int(3.0 / model.opt.timestep):
                    data.eq_active[weld_id] = 0
                    SLITTA_LOCKED           = False

                    model.dof_frictionloss[door_dofadr] = FRICTION_STATIC

                    q_interp_start      = q_des_all.copy()
                    slitta_interp_start = float(data.qpos[slitta_qposadr])
                    reset_integral()
                    phase       = RETURN_NEUTRAL
                    phase_timer = 0

            # FASE 8: RETURN TO NEUTRAL
            # Translate chassis back to neutral while interpolating arm to
            # Q_NEUTRAL. Advances to the next door or ends the cycle.
            case 8:
                set_joint_impedance(K_APPROACH, D_APPROACH, JOINT_FORCE_LIMITS)

                slitta_pos = data.qpos[slitta_qposadr]
                slitta_vel = data.qvel[slitta_dofadr]

                q_des_all[:] = interpolate_arm(q_interp_start, Q_NEUTRAL, slitta_interp_start, SLITTA_NEUTRAL)

                data.ctrl[ACT_SLITTA] = SLITTA_NEUTRAL

                if abs(slitta_pos - SLITTA_NEUTRAL) < 0.01 and abs(slitta_vel) < 0.01:
                    data.qpos[slitta_qposadr] = SLITTA_NEUTRAL
                    data.qvel[slitta_dofadr]  = 0.0

                    q_des_all[:] = Q_NEUTRAL.copy()
                    error_integral[:] = 0.0
                    data.qvel[:] = 0.0
                    data.qacc[:] = 0.0
                    mujoco.mj_forward(model, data)

                    door_index += 1
                    if door_index >= len(DOOR_SEQUENCE):
                        door_index   = 0
                        cycle_count += 1

                    if cycle_count >= N_CYCLES:
                        phase = DONE
                        print(f"\n✓ Completati {N_CYCLES} cicli.")
                    else:
                        next_key = DOOR_SEQUENCE[door_index]
                        print(f"\n→ Ciclo {cycle_count + 1}/{N_CYCLES} | porta: {door_configs[next_key]['label']}")
                        setup_door(next_key)

                        q_interp_start      = Q_NEUTRAL.copy()
                        slitta_interp_start = SLITTA_NEUTRAL
                        open_start_time     = None
                        close_start_time    = None
                        phase           = MOVE_TO_DOOR
                        phase_timer     = 0
                        traj_index      = 1
                        close_index     = 0
                        best_dist       = float("inf")
                        stall_counter   = 0

        if phase != DONE:

            if SLITTA_LOCKED:
                data.qpos[slitta_qposadr] = SLITTA_LOCK_POS
                data.qvel[slitta_dofadr]  = 0.0
                data.ctrl[ACT_SLITTA]     = SLITTA_LOCK_POS
                mujoco.mj_forward(model, data)

            joint_torques = data.qfrc_actuator[all_dofadr]

            log_time.append(step * model.opt.timestep)
            log_door_deg.append(np.rad2deg(data.qpos[door_qposadr]))
            log_door_vel.append(np.rad2deg(data.qvel[door_dofadr]))
            log_tau.append(joint_torques.copy())
            log_porta.append(0 if active.get("label") == "SINISTRA" else 1)
            log_slitta.append(data.qpos[slitta_qposadr])

            data.ctrl[ACT_ARM] = q_des_all

            mujoco.mj_step(model, data)
            viewer.sync()

            if step % 50 == 0:
                door_angle    = np.rad2deg(data.qpos[door_qposadr])
                door_vel      = np.rad2deg(data.qvel[door_dofadr])
                t             = step * model.opt.timestep
                lbl           = active.get("label", "?")
                spos          = data.qpos[slitta_qposadr]
                joint_torques = data.qfrc_actuator[all_dofadr]
                violations    = np.abs(joint_torques) > JOINT_FORCE_LIMITS

                print(
                    f"t={t:6.2f}s | ciclo {cycle_count + 1}/{N_CYCLES} "
                    f"| porta {lbl:>8s} | {STATE_NAMES[phase]:>12s} "
                    f"| slitta={spos:+.3f} | porta={door_angle:6.1f}° "
                    f"| vel={door_vel:+.1f}°/s "
                    f"| tau={np.round(joint_torques, 1)} "
                    f"| {'*** VIOLAZIONE J' + str(np.where(violations)[0] + 1) + ' ***' if violations.any() else 'ok'}"
                )

print("Simulazione terminata.")
scipy.io.savemat("dati_porta.mat", {
    "tempo":    np.array(log_time),
    "apertura": np.array(log_door_deg),
    "velocita": np.array(log_door_vel),
    "tau":      np.array(log_tau),
    "porta":    np.array(log_porta),
    "slitta":   np.array(log_slitta),
})