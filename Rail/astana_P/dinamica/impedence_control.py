import mujoco
import mujoco.viewer
import numpy as np

# ══════════════════════════════════════════════════════════════════
#  PARAMETRI PRINCIPALI
# ══════════════════════════════════════════════════════════════════
N_CYCLES = 2
OPEN_DEG = 62

XML_PATH     = "/Users/cristianvoltan/Desktop/unipd/tirocinio/CAD/Rail/astana_P/Astana_p.xml"
TRAJ_SX_PATH = "/Users/cristianvoltan/Desktop/unipd/tirocinio/CAD/Rail/astana_P/traiettorie/traiettoria_sinistra.npy"
TRAJ_DX_PATH = "/Users/cristianvoltan/Desktop/unipd/tirocinio/CAD/Rail/astana_P/traiettorie/traiettoria_destra.npy"

# ══════════════════════════════════════════════════════════════════
#  PARAMETRI IMPEDANCE CONTROL
# ══════════════════════════════════════════════════════════════════
IMP_KP = np.array([600.0, 600.0, 500.0, 500.0, 1000.0, 1000.0, 600.0])
IMP_KD = np.array([ 65.0,  65.0,  50.0,  50.0,   80.0,   80.0,  60.0])

# ← FISSO J6/J7 — guadagni altissimi su joint6 e joint7 per tenerli bloccati
IMP_KP[5] = 3000.0
IMP_KP[6] = 3000.0
IMP_KD[5] =  200.0
IMP_KD[6] =  200.0

IMP_FORCE_LIMIT = np.array([150.0, 150.0, 150.0, 150.0, 150.0, 150.0, 150.0])

SLITTA_KP = 3000.0
SLITTA_KD = 2500.0


# ══════════════════════════════════════════════════════════════════
#  CARICAMENTO MODELLO
# ══════════════════════════════════════════════════════════════════
model = mujoco.MjModel.from_xml_path(XML_PATH)
data  = mujoco.MjData(model)

ACT_SLITTA  = 0
ACT_ARM     = [1, 2, 3, 4, 5, 6, 7]
ACT_GRIPPER = 8

for i, a in enumerate(ACT_ARM):
    model.actuator_gainprm[a, 0]    = 1.0
    model.actuator_biastype[a]      = 0
    model.actuator_biasprm[a, :]    = 0.0
    model.actuator_ctrlrange[a, 0]  = -IMP_FORCE_LIMIT[i]
    model.actuator_ctrlrange[a, 1]  =  IMP_FORCE_LIMIT[i]
    model.actuator_forcerange[a, 0] = -IMP_FORCE_LIMIT[i]
    model.actuator_forcerange[a, 1] =  IMP_FORCE_LIMIT[i]


# ══════════════════════════════════════════════════════════════════
#  JOINT DEL BRACCIO
# ══════════════════════════════════════════════════════════════════
all_joint_names = ["joint1", "joint2", "joint3", "joint4", "joint5", "joint6", "joint7"]

# ← FISSO J6/J7 — l'IK agisce solo sui primi 5 giunti
ik_joint_names  = ["joint1", "joint2", "joint3", "joint4", "joint5"]

all_qposadr, all_dofadr = [], []
for name in all_joint_names:
    jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)
    if jid == -1:
        raise ValueError(f"Joint non trovato nel modello: {name}")
    all_qposadr.append(model.jnt_qposadr[jid])
    all_dofadr.append(model.jnt_dofadr[jid])

all_qposadr = np.array(all_qposadr)
all_dofadr  = np.array(all_dofadr)

# ← FISSO J6/J7 — indirizzi IK solo per i primi 5
ik_qposadr = all_qposadr[:5]
ik_dofadr  = all_dofadr[:5]


# ══════════════════════════════════════════════════════════════════
#  SLITTA
# ══════════════════════════════════════════════════════════════════
slitta_joint_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "giunto8_slitta")
if slitta_joint_id == -1:
    raise ValueError("Joint della slitta 'giunto8_slitta' non trovato nel modello.")

slitta_qposadr = model.jnt_qposadr[slitta_joint_id]
slitta_dofadr  = model.jnt_dofadr[slitta_joint_id]

SLITTA_NEUTRAL = 0.0
SLITTA_POS_SX  = -0.05
SLITTA_POS_DX  = 0.71


# ══════════════════════════════════════════════════════════════════
#  POSIZIONE NEUTRA BRACCIO
# ══════════════════════════════════════════════════════════════════
Q_NEUTRAL = np.array([0.0, -1.04, 0.0, -2.46, 0.0, 2.73, 0.811])

# ← FISSO J6/J7 — valori fissi di joint6 e joint7 (presi da Q_NEUTRAL)
Q_J6_FIXED = Q_NEUTRAL[5]   # 2.73 rad
Q_J7_FIXED = Q_NEUTRAL[6]   # 0.811 rad


# ══════════════════════════════════════════════════════════════════
#  TRAIETTORIE
# ══════════════════════════════════════════════════════════════════
def build_traj_close(traj, open_deg, max_deg=95):
    n = max(2, min(int(len(traj) * open_deg / max_deg), len(traj)))
    return traj[:n][::-1].copy(), n

trajectory_sx = np.load(TRAJ_SX_PATH)
trajectory_dx = np.load(TRAJ_DX_PATH)

traj_close_sx, n_open_sx = build_traj_close(trajectory_sx, OPEN_DEG)
traj_close_dx, n_open_dx = build_traj_close(trajectory_dx, OPEN_DEG)


# ══════════════════════════════════════════════════════════════════
#  CONFIGURAZIONI PER PORTA
# ← FISSO J6/J7 — q_home con joint6 e joint7 bloccati ai valori fissi
# ══════════════════════════════════════════════════════════════════
door_configs = {
    "sx": dict(
        label          = "SINISTRA",
        slitta_target  = SLITTA_POS_SX,
        q_home         = np.array([0.956, -0.053, 0.0, -1.48, 0.0, Q_J6_FIXED, Q_J7_FIXED]),
        site_name      = "grasp_site",
        target_site    = "target_manigliasx",
        weld_name      = "presa_sx",
        door_joint     = "giunto_porta_sx",
        trajectory     = trajectory_sx,
        traj_close     = traj_close_sx,
        n_open         = n_open_sx,
    ),
    "dx": dict(
        label          = "DESTRA",
        slitta_target  = SLITTA_POS_DX,
        q_home         = np.array([0.956, -0.053, 0.0, -1.48, 0.0, Q_J6_FIXED, Q_J7_FIXED]),
        site_name      = "grasp_site",
        target_site    = "target_manigliadx",
        weld_name      = "presa_dx",
        door_joint     = "giunto_porta_dx",
        trajectory     = trajectory_dx,
        traj_close     = traj_close_dx,
        n_open         = n_open_dx,
    ),
}

DOOR_SEQUENCE = ["sx", "dx"]


# ══════════════════════════════════════════════════════════════════
#  STATE MACHINE
# ══════════════════════════════════════════════════════════════════
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


# ══════════════════════════════════════════════════════════════════
#  PARAMETRI IK
# ══════════════════════════════════════════════════════════════════
KP_IK         = 1.00
KI_IK         = 0.15
INTEGRAL_CLIP = 0.05
damping       = 5e-3
ik_every      = 1

gain_phase0 = 0.005
gain_phase3 = 0.030
gain_close  = 0.030

CLOSE_DURATION          = 300
TRAJ_WAIT               = 2
STALL_LIMIT             = 2000
WAIT_BEFORE_CLOSE_STEPS = int(13.0 / model.opt.timestep)


# ══════════════════════════════════════════════════════════════════
#  VARIABILI GLOBALI RUNTIME
# ══════════════════════════════════════════════════════════════════
error_integral = np.zeros(3)

q_des_all  = np.zeros(len(all_joint_names))
q_des_ik   = np.zeros(len(ik_joint_names))   # ← FISSO J6/J7: ora size=5
q_des_prev = np.zeros(len(all_joint_names))

site_id        = None
target_site_id = None
weld_id        = None
door_qposadr   = None
active         = {}

SLITTA_LOCKED   = False
SLITTA_LOCK_POS = None

q_interp_start      = Q_NEUTRAL.copy()
slitta_interp_start = SLITTA_NEUTRAL

_M_full = np.zeros((model.nv, model.nv))


# ══════════════════════════════════════════════════════════════════
#  HELPER: forza joint6 e joint7 ai valori fissi in q_des_all
# ← FISSO J6/J7 — chiamato ogni volta che q_des_all viene aggiornato
# ══════════════════════════════════════════════════════════════════
def lock_j6_j7():
    q_des_all[5] = Q_J6_FIXED
    q_des_all[6] = Q_J7_FIXED


# ══════════════════════════════════════════════════════════════════
#  IMPEDANCE CONTROLLER
# ══════════════════════════════════════════════════════════════════
def compute_impedance_torques(q_des, dq_des=None):
    if dq_des is None:
        dq_des = np.zeros(len(all_joint_names))

    q_cur  = data.qpos[all_qposadr]
    dq_cur = data.qvel[all_dofadr]

    q_err  = q_des  - q_cur
    dq_err = dq_des - dq_cur

    mujoco.mj_fullM(model, _M_full, data.qM)
    M    = _M_full[np.ix_(all_dofadr, all_dofadr)]
    bias = data.qfrc_bias[all_dofadr]

    virtual_acc = IMP_KP * q_err + IMP_KD * dq_err
    tau_ctrl    = M @ virtual_acc
    tau_ctrl    = np.clip(tau_ctrl, -IMP_FORCE_LIMIT, IMP_FORCE_LIMIT)

    return tau_ctrl + bias


# ══════════════════════════════════════════════════════════════════
#  FUNZIONI DI SUPPORTO
# ══════════════════════════════════════════════════════════════════
def reset_integral():
    global error_integral
    error_integral[:] = 0.0


def setup_door(key):
    global active, error_integral
    global site_id, target_site_id, weld_id, door_qposadr

    cfg = door_configs[key]
    active = cfg

    site_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, cfg["site_name"])
    target_site_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, cfg["target_site"])
    weld_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_EQUALITY, cfg["weld_name"])
    door_joint_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, cfg["door_joint"])

    if any(x == -1 for x in [site_id, target_site_id, weld_id, door_joint_id]):
        raise ValueError(f"Riferimento non trovato per porta '{key}'.")

    door_qposadr = model.jnt_qposadr[door_joint_id]
    reset_integral()


def interpolate_arm(q_start, q_end, slitta_start, slitta_target):
    slitta_pos   = data.qpos[slitta_qposadr]
    total_travel = abs(slitta_target - slitta_start)
    traveled     = abs(slitta_pos - slitta_start)
    alpha        = np.clip(traveled / total_travel, 0.0, 1.0) if total_travel > 1e-6 else 1.0

    q_des_all[:] = (1.0 - alpha) * q_start + alpha * q_end

    # ← FISSO J6/J7 — sovrascrive sempre i valori interpolati di j6/j7
    lock_j6_j7()
    return alpha


def solve_position_ik(target, gain):
    """
    IK differenziale agisce solo su joint1–joint5.
    joint6 e joint7 non vengono toccati.
    """
    global q_des_ik, error_integral

    site_pos  = data.site_xpos[site_id].copy()
    error_pos = target - site_pos

    error_integral += error_pos * model.opt.timestep
    error_integral  = np.clip(error_integral, -INTEGRAL_CLIP, INTEGRAL_CLIP)

    control = KP_IK * error_pos + KI_IK * error_integral

    jacp = np.zeros((3, model.nv))
    mujoco.mj_jacSite(model, data, jacp, None, site_id)

    # ← FISSO J6/J7 — Jacobiana ristretta ai soli dof IK (joint1–5)
    J = jacp[:, ik_dofadr]

    A  = J @ J.T + damping * np.eye(3)
    dq = J.T @ np.linalg.solve(A, control)

    q_des_ik += gain * dq

    for i, name in enumerate(ik_joint_names):
        jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)
        if model.jnt_limited[jid]:
            lo, hi = model.jnt_range[jid]
            q_des_ik[i] = np.clip(q_des_ik[i], lo, hi)

    # ← FISSO J6/J7 — copia solo i primi 5 in q_des_all, j6/j7 restano fissi
    q_des_all[:5] = q_des_ik
    lock_j6_j7()

    return np.linalg.norm(error_pos)


# ══════════════════════════════════════════════════════════════════
#  STATO INIZIALE
# ══════════════════════════════════════════════════════════════════
for i in range(len(all_joint_names)):
    data.qpos[all_qposadr[i]] = Q_NEUTRAL[i]

data.qpos[slitta_qposadr] = SLITTA_NEUTRAL
data.ctrl[ACT_GRIPPER]    = 255
data.ctrl[ACT_SLITTA]     = SLITTA_NEUTRAL

mujoco.mj_forward(model, data)

q_des_all[:]  = Q_NEUTRAL.copy()
q_des_ik[:]   = Q_NEUTRAL[:5].copy()   # ← FISSO J6/J7: solo 5 elementi
q_des_prev[:] = Q_NEUTRAL.copy()

lock_j6_j7()   # ← FISSO J6/J7: inizializzazione esplicita

tau0 = compute_impedance_torques(Q_NEUTRAL)
for i, a in enumerate(ACT_ARM):
    data.ctrl[a] = tau0[i]


# ══════════════════════════════════════════════════════════════════
#  INIZIALIZZAZIONE CICLO
# ══════════════════════════════════════════════════════════════════
door_index  = 0
cycle_count = 0

setup_door(DOOR_SEQUENCE[door_index])

q_interp_start      = Q_NEUTRAL.copy()
slitta_interp_start = SLITTA_NEUTRAL

phase           = MOVE_TO_DOOR
phase_timer     = 0
traj_index      = 1
close_index     = 0
traj_wait_timer = 0
best_dist       = float("inf")
stall_counter   = 0

MAX_STEPS = int(40000 * N_CYCLES * len(DOOR_SEQUENCE)) + 10000


# ══════════════════════════════════════════════════════════════════
#  LOOP DI SIMULAZIONE
# ══════════════════════════════════════════════════════════════════
with mujoco.viewer.launch_passive(model, data) as viewer:

    for step in range(MAX_STEPS):

        if phase == DONE:
            break

        mujoco.mj_forward(model, data)

        match phase:

            case 0:
                slitta_tgt = active["slitta_target"]
                slitta_pos = data.qpos[slitta_qposadr]
                slitta_vel = data.qvel[slitta_dofadr]

                interpolate_arm(q_interp_start, active["q_home"],
                                 slitta_interp_start, slitta_tgt)

                data.ctrl[ACT_SLITTA] = slitta_tgt

                if abs(slitta_pos - slitta_tgt) < 0.01 and abs(slitta_vel) < 0.01:
                    SLITTA_LOCKED   = True
                    SLITTA_LOCK_POS = slitta_tgt

                    data.qpos[slitta_qposadr] = SLITTA_LOCK_POS
                    data.qvel[slitta_dofadr]  = 0.0

                    q_des_all[:]  = active["q_home"].copy()
                    q_des_ik[:]   = active["q_home"][:5].copy()  # ← FISSO J6/J7
                    q_des_prev[:] = q_des_all.copy()
                    lock_j6_j7()                                  # ← FISSO J6/J7

                    mujoco.mj_forward(model, data)
                    reset_integral()

                    phase         = APPROACH_HANDLE
                    phase_timer   = 0
                    best_dist     = float("inf")
                    stall_counter = 0

            case 1:
                if step % ik_every == 0:
                    target_pos = active["trajectory"][0, 0:3]
                    dist = solve_position_ik(target_pos, gain_phase0)
                    # q_des_all già aggiornato dentro solve_position_ik

                    if dist < best_dist - 0.001:
                        best_dist     = dist
                        stall_counter = 0
                    else:
                        stall_counter += 1

                    if dist < 0.035 or stall_counter >= STALL_LIMIT:
                        reset_integral()
                        phase       = CLOSE_GRIPPER
                        phase_timer = 0

            case 2:
                phase_timer += 1
                if phase_timer >= CLOSE_DURATION:
                    data.qvel[:] = 0.0
                    data.qacc[:] = 0.0
                    data.eq_active[weld_id] = 1
                    mujoco.mj_forward(model, data)
                    reset_integral()
                    phase       = STABILIZE_WELD
                    phase_timer = 0

            case 3:
                phase_timer += 1
                data.eq_active[weld_id] = 1
                if phase_timer >= 100:
                    traj_index      = 1
                    traj_wait_timer = 0
                    q_des_prev[:]   = q_des_all.copy()
                    reset_integral()
                    phase = OPEN_DOOR

            case 4:
                if step % ik_every == 0:
                    data.eq_active[weld_id] = 1
                    target_pos = active["trajectory"][traj_index, 0:3]
                    dist = solve_position_ik(target_pos, gain_phase3)

                    if dist < 0.03:
                        traj_wait_timer += 1
                        if traj_wait_timer >= TRAJ_WAIT and traj_index < active["n_open"] - 1:
                            traj_index += 1
                            traj_wait_timer = 0
                            reset_integral()
                    else:
                        traj_wait_timer = 0

                    if traj_index >= active["n_open"] - 1 and dist < 0.02:
                        reset_integral()
                        phase       = WAIT_OPEN
                        phase_timer = 0

            case 5:
                data.eq_active[weld_id] = 1
                phase_timer += 1
                if phase_timer >= WAIT_BEFORE_CLOSE_STEPS:
                    data.qvel[:] = 0.0
                    data.qacc[:] = 0.0
                    mujoco.mj_forward(model, data)
                    close_index     = 0
                    traj_wait_timer = 0
                    q_des_prev[:]   = q_des_all.copy()
                    reset_integral()
                    phase = CLOSE_DOOR

            case 6:
                if step % ik_every == 0:
                    data.eq_active[weld_id] = 1
                    target_pos = active["traj_close"][close_index, 0:3]
                    dist = solve_position_ik(target_pos, gain_close)

                    if dist < 0.03:
                        traj_wait_timer += 1
                        if traj_wait_timer >= TRAJ_WAIT and close_index < len(active["traj_close"]) - 1:
                            close_index += 1
                            traj_wait_timer = 0
                            reset_integral()
                    else:
                        traj_wait_timer = 0

                    if close_index >= len(active["traj_close"]) - 1 and dist < 0.002:
                        reset_integral()
                        phase       = HOLD_CLOSED
                        phase_timer = 0

            case 7:
                data.eq_active[weld_id] = 1
                phase_timer += 1
                if phase_timer >= int(3.0 / model.opt.timestep):
                    data.eq_active[weld_id] = 0
                    SLITTA_LOCKED           = False
                    q_interp_start          = q_des_all.copy()
                    slitta_interp_start     = float(data.qpos[slitta_qposadr])
                    q_des_prev[:]           = q_des_all.copy()
                    reset_integral()
                    phase       = RETURN_NEUTRAL
                    phase_timer = 0

            case 8:
                slitta_pos = data.qpos[slitta_qposadr]
                slitta_vel = data.qvel[slitta_dofadr]

                interpolate_arm(q_interp_start, Q_NEUTRAL,
                                 slitta_interp_start, SLITTA_NEUTRAL)

                data.ctrl[ACT_SLITTA] = SLITTA_NEUTRAL

                if abs(slitta_pos - SLITTA_NEUTRAL) < 0.01 and abs(slitta_vel) < 0.01:
                    data.qpos[slitta_qposadr] = SLITTA_NEUTRAL
                    data.qvel[slitta_dofadr]  = 0.0

                    q_des_all[:]  = Q_NEUTRAL.copy()
                    q_des_ik[:]   = Q_NEUTRAL[:5].copy()  # ← FISSO J6/J7
                    q_des_prev[:] = Q_NEUTRAL.copy()
                    lock_j6_j7()                           # ← FISSO J6/J7

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
                        phase           = MOVE_TO_DOOR
                        phase_timer     = 0
                        traj_index      = 1
                        close_index     = 0
                        traj_wait_timer = 0
                        best_dist       = float("inf")
                        stall_counter   = 0

        # ══════════════════════════════════════════════════════════
        #  APPLICAZIONE CONTROLLO DI IMPEDENZA
        # ══════════════════════════════════════════════════════════
        if phase != DONE:

            if SLITTA_LOCKED:
                data.qpos[slitta_qposadr] = SLITTA_LOCK_POS
                data.qvel[slitta_dofadr]  = 0.0
                data.ctrl[ACT_SLITTA]     = SLITTA_LOCK_POS

            # ← FISSO J6/J7 — garantisce che q_des_all non abbia derive
            lock_j6_j7()

            dq_des_ff = (q_des_all - q_des_prev) / model.opt.timestep
            tau       = compute_impedance_torques(q_des_all, dq_des_ff)

            for i, a in enumerate(ACT_ARM):
                data.ctrl[a] = tau[i]

            q_des_prev[:] = q_des_all.copy()

            open_gripper_phases = (MOVE_TO_DOOR, APPROACH_HANDLE, RETURN_NEUTRAL)
            data.ctrl[ACT_GRIPPER] = 255 if phase in open_gripper_phases else 50

            mujoco.mj_step(model, data)
            viewer.sync()

            if step % 50 == 0:
                door_angle = np.rad2deg(data.qpos[door_qposadr])
                t          = step * model.opt.timestep
                lbl        = active.get("label", "?")
                spos       = data.qpos[slitta_qposadr]
                tau_norm   = np.linalg.norm(tau)
                j6_err     = np.rad2deg(abs(data.qpos[all_qposadr[5]] - Q_J6_FIXED))
                j7_err     = np.rad2deg(abs(data.qpos[all_qposadr[6]] - Q_J7_FIXED))

                print(
                    f"t={t:6.2f}s | ciclo {cycle_count + 1}/{N_CYCLES} "
                    f"| porta {lbl:>8s} | {STATE_NAMES[phase]:>12s} "
                    f"| slitta={spos:+.3f} | porta={door_angle:6.1f}° "
                    f"| |τ|={tau_norm:6.1f} N·m "
                    f"| j6_err={j6_err:.2f}° j7_err={j7_err:.2f}°"  # ← FISSO J6/J7
                )

print("Simulazione terminata.")