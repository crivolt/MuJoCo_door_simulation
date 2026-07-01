import mujoco
import mujoco.viewer
import numpy as np

# ══════════════════════════════════════════════════════════════════
#  PARAMETRI PRINCIPALI
# ══════════════════════════════════════════════════════════════════
N_CYCLES = 2

XML_PATH      = "/Users/cristianvoltan/Desktop/unipd/tirocinio/CAD/Rail/osaka/osaka.xml"
TRAJ_SX_PATH  = "/Users/cristianvoltan/Desktop/unipd/tirocinio/CAD/Rail/osaka/traiettorie/traiettoria_sinistra.npy"
TRAJ_DX_PATH  = "/Users/cristianvoltan/Desktop/unipd/tirocinio/CAD/Rail/osaka/traiettorie/traiettoria_destra.npy"

# ══════════════════════════════════════════════════════════════════
#  CARICAMENTO MODELLO
# ══════════════════════════════════════════════════════════════════
model = mujoco.MjModel.from_xml_path(XML_PATH)
data  = mujoco.MjData(model)

# ══════════════════════════════════════════════════════════════════
#  INDICI ATTUATORI
#  (layout XML identico ad athen_xl con slitta)
#   0 → act_slitta    (giunto8_slitta)
#   1-7 → actuator1..7  (braccio Panda)
#   8 → actuator8    (gripper / tendon split)
# ══════════════════════════════════════════════════════════════════
ACT_SLITTA  = 0
ACT_ARM     = list(range(1, 8))   # [1,2,3,4,5,6,7]
ACT_GRIPPER = 8

# ══════════════════════════════════════════════════════════════════
#  PARAMETRI IMPEDENZA NELLO SPAZIO DEI GIUNTI
#
#  Legge di controllo:
#      τ = Kp · (q_d − q) − Kd · dq + g(q)
#
#  Profilo FREE    : moto slitta / ritorno neutro
#  Profilo CONTACT : approccio maniglia / apertura-chiusura porta
# ══════════════════════════════════════════════════════════════════
#                          j1     j2     j3     j4     j5    j6    j7
KP_FREE    = np.diag([500., 500., 400., 400., 300.,  80.,  40.])
KD_FREE    = np.diag([ 50.,  50.,  40.,  40.,  30.,   8.,   4.])

KP_CONTACT = np.diag([200., 200., 150., 150., 120.,  40.,  20.])
KD_CONTACT = np.diag([ 40.,  40.,  30.,  30.,  24.,   6.,   3.])

FORCE_LIMITS = np.array([87., 87., 87., 87., 12., 12., 12.])   # [N·m]

CONTACT_PHASES = {1, 2, 3, 4, 5, 6, 7, 8}   # tutte tranne MOVE_SLITTA (0) e RETURN (9)

# ══════════════════════════════════════════════════════════════════
#  GAINPRM / BIASPRM2 — letti dall'XML per l'inversione
#  f_i = gainprm_i*(ctrl_i − q_i) + biasprm2_i*dq_i
#  → ctrl_i = q_i + (τ_i − biasprm2_i*dq_i) / gainprm_i
# ══════════════════════════════════════════════════════════════════
arm_gainprm  = np.array([model.actuator_gainprm[i, 0] for i in ACT_ARM])
arm_biasprm2 = np.array([model.actuator_biasprm[i, 2] for i in ACT_ARM])
assert np.all(arm_gainprm != 0), "gainprm=0 trovato in un attuatore del braccio!"

# ══════════════════════════════════════════════════════════════════
#  JOINT DEL BRACCIO
# ══════════════════════════════════════════════════════════════════
all_joint_names = ["joint1","joint2","joint3","joint4","joint5","joint6","joint7"]
ik_joint_names  = ["joint1","joint2","joint3","joint4","joint5"]   # j6/j7 fissi

all_qposadr, all_dofadr = [], []
for name in all_joint_names:
    jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)
    if jid == -1:
        raise ValueError(f"Joint non trovato: {name}")
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

# ══════════════════════════════════════════════════════════════════
#  SLITTA
# ══════════════════════════════════════════════════════════════════
slitta_jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "giunto8_slitta")
if slitta_jid == -1:
    raise ValueError("Joint 'giunto8_slitta' non trovato.")
slitta_qposadr = model.jnt_qposadr[slitta_jid]
slitta_dofadr  = model.jnt_dofadr[slitta_jid]

SLITTA_NEUTRAL = 0.00
SLITTA_TOL     = 0.05
SLITTA_VEL_TOL = 0.05

# ══════════════════════════════════════════════════════════════════
#  GIUNTI FISSI J6 / J7
# ══════════════════════════════════════════════════════════════════
J6_IDX   = all_joint_names.index("joint6")
J7_IDX   = all_joint_names.index("joint7")
J6_FIXED = 2.77
J7_FIXED = 0.927

def lock_fixed_joints():
    q_des_all[J6_IDX] = J6_FIXED
    q_des_all[J7_IDX] = J7_FIXED

def hard_lock_fixed_joints():
    """Azzera deriva fisica su j6/j7."""
    for idx in (J6_IDX, J7_IDX):
        data.qpos[all_qposadr[idx]] = q_des_all[idx]
        data.qvel[all_dofadr[idx]]  = 0.0

# ══════════════════════════════════════════════════════════════════
#  POSIZIONE NEUTRA
# ══════════════════════════════════════════════════════════════════
Q_NEUTRAL = np.array([0.0, -0.335, 0.0, -2.19, 0.0, J6_FIXED, J7_FIXED])

# ══════════════════════════════════════════════════════════════════
#  TRAIETTORIE
# ══════════════════════════════════════════════════════════════════
trajectory_sx = np.load(TRAJ_SX_PATH)
trajectory_dx = np.load(TRAJ_DX_PATH)

OPEN_METERS = 0.7
MAX_METERS  = 0.8   # range porta osaka (0-0.8 m)

def build_trajectories(traj_full):
    n_open = int(len(traj_full) * OPEN_METERS / MAX_METERS)
    n_open = max(2, min(n_open, len(traj_full)))
    traj_open  = traj_full[:n_open]
    traj_close = traj_open[::-1].copy()
    return traj_open, traj_close, n_open

traj_sx_open, traj_sx_close, n_open_sx = build_trajectories(trajectory_sx)
traj_dx_open, traj_dx_close, n_open_dx = build_trajectories(trajectory_dx)

# ══════════════════════════════════════════════════════════════════
#  CONFIGURAZIONI PORTA
#  ← CALIBRA slitta_at_handle e slitta_door_open sul tuo setup
# ══════════════════════════════════════════════════════════════════
door_configs = {
    "sx": dict(
        label            = "SINISTRA",
        slitta_at_handle = -0.89,   # ← CALIBRA
        slitta_door_open = -0.19,   # ← slitta arretra → porta si apre
        q_home           = Q_NEUTRAL.copy(),
        traj_open        = traj_sx_open,
        traj_close       = traj_sx_close,
        n_open           = n_open_sx,
        site_name        = "grasp_site",
        target_site      = "target_manigliasx",
        weld_name        = "presa_sx",
        door_joint       = "slide_sx",
    ),
    "dx": dict(
        label            = "DESTRA",
        slitta_at_handle =  0.84,   # ← CALIBRA
        slitta_door_open =  0.17,   # ← slitta avanza → porta si apre
        q_home           = Q_NEUTRAL.copy(),
        traj_open        = traj_dx_open,
        traj_close       = traj_dx_close,
        n_open           = n_open_dx,
        site_name        = "grasp_site",
        target_site      = "target_manigliadx",
        weld_name        = "presa_dx",
        door_joint       = "slide_dx",
    ),
}

DOOR_SEQUENCE = ["sx", "dx"]

# ══════════════════════════════════════════════════════════════════
#  STATE MACHINE
# ══════════════════════════════════════════════════════════════════
MOVE_SLITTA_TO_HANDLE = 0
APPROACH_HANDLE       = 1
CLOSE_GRIPPER         = 2
STABILIZE_WELD        = 3
OPEN_DOOR             = 4
WAIT_OPEN             = 5
CLOSE_DOOR            = 6
HOLD_CLOSED           = 7
RETURN_NEUTRAL        = 8
DONE                  = 9

STATE_NAMES = {
    0: "MOVE_SLITTA ",
    1: "APPROACH    ",
    2: "CLOSE_GRIP  ",
    3: "STABILIZE   ",
    4: "OPEN_DOOR   ",
    5: "WAIT_OPEN   ",
    6: "CLOSE_DOOR  ",
    7: "HOLD_CLOSED ",
    8: "RETURN      ",
    9: "DONE        ",
}

# ══════════════════════════════════════════════════════════════════
#  PARAMETRI IK
# ══════════════════════════════════════════════════════════════════
KP_IK         = 1.0
KI_IK         = 0.15
INTEGRAL_CLIP = 0.08
damping       = 5e-3
ik_every      = 1

gain_approach = 0.002
gain_tracking = 0.035

DIST_APPROACH = 0.01    # distanza per chiudere il gripper [m]
DIST_TRAJ     = 0.03    # distanza per avanzare al waypoint successivo [m]
DIST_FINAL    = 0.02    # distanza per considerare traiettoria completata [m]

TRAJ_WAIT    = 2        # step di attesa tra un waypoint e il successivo
STALL_LIMIT  = 2000

CLOSE_DURATION  = 300
STABILIZE_STEPS = 100
WAIT_OPEN_STEPS = int(13.0 / model.opt.timestep)
HOLD_STEPS      = int(8.0  / model.opt.timestep)

# ══════════════════════════════════════════════════════════════════
#  VARIABILI GLOBALI RUNTIME
# ══════════════════════════════════════════════════════════════════
error_integral = np.zeros(3)
q_des_all      = Q_NEUTRAL.copy()
q_des_ik       = Q_NEUTRAL[:5].copy()   # j1..j5

active         = {}
site_id        = None
target_site_id = None
weld_id        = None
door_qposadr   = None

q_interp_start      = Q_NEUTRAL.copy()
slitta_interp_start = SLITTA_NEUTRAL

# ══════════════════════════════════════════════════════════════════
#  STATO INIZIALE
# ══════════════════════════════════════════════════════════════════
for i, adr in enumerate(all_qposadr):
    data.qpos[adr] = Q_NEUTRAL[i]
data.qpos[slitta_qposadr] = SLITTA_NEUTRAL
data.ctrl[ACT_SLITTA]     = SLITTA_NEUTRAL
data.ctrl[ACT_GRIPPER]    = 255
mujoco.mj_forward(model, data)
lock_fixed_joints()

# ══════════════════════════════════════════════════════════════════
#  HELPER
# ══════════════════════════════════════════════════════════════════
def reset_integral():
    global error_integral
    error_integral[:] = 0.0


def slitta_reached(target):
    pos = data.qpos[slitta_qposadr]
    vel = data.qvel[slitta_dofadr]
    return abs(pos - target) < SLITTA_TOL and abs(vel) < SLITTA_VEL_TOL


def setup_door(key):
    global active, site_id, target_site_id, weld_id, door_qposadr
    cfg            = door_configs[key]
    active         = cfg
    site_id        = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE,     cfg["site_name"])
    target_site_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE,     cfg["target_site"])
    weld_id        = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_EQUALITY, cfg["weld_name"])
    door_jid_      = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT,    cfg["door_joint"])
    if any(x == -1 for x in [site_id, target_site_id, weld_id, door_jid_]):
        raise ValueError(f"Riferimento non trovato per porta '{key}'.")
    door_qposadr = model.jnt_qposadr[door_jid_]
    reset_integral()


def interpolate_arm(q_start, q_end, slitta_start, slitta_target):
    """Interpola q_des_all linearmente rispetto alla posizione slitta."""
    slitta_pos   = data.qpos[slitta_qposadr]
    total_travel = abs(slitta_target - slitta_start)
    traveled     = abs(slitta_pos   - slitta_start)
    alpha        = np.clip(traveled / total_travel, 0.0, 1.0) if total_travel > 1e-6 else 1.0
    q_des_all[:] = (1.0 - alpha) * q_start + alpha * q_end
    lock_fixed_joints()
    return alpha


# ══════════════════════════════════════════════════════════════════
#  IK DIFFERENZIALE (PI) — aggiorna q_des_all, non applica ctrl
# ══════════════════════════════════════════════════════════════════
def solve_position_ik(target: np.ndarray, gain: float) -> float:
    global q_des_ik, error_integral

    mujoco.mj_forward(model, data)
    site_pos  = data.site_xpos[site_id].copy()
    error_pos = target - site_pos

    error_integral += error_pos * model.opt.timestep
    error_integral  = np.clip(error_integral, -INTEGRAL_CLIP, INTEGRAL_CLIP)
    control = KP_IK * error_pos + KI_IK * error_integral

    jacp = np.zeros((3, model.nv))
    mujoco.mj_jacSite(model, data, jacp, None, site_id)
    J  = jacp[:, ik_dofadr]                        # (3, 5)
    A  = J @ J.T + damping * np.eye(3)
    dq = J.T @ np.linalg.solve(A, control)
    q_des_ik += gain * dq

    for i, name in enumerate(ik_joint_names):
        jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)
        if model.jnt_limited[jid]:
            lo, hi = model.jnt_range[jid]
            q_des_ik[i] = np.clip(q_des_ik[i], lo, hi)

    q_des_all[:5] = q_des_ik
    lock_fixed_joints()
    return float(np.linalg.norm(error_pos))


# ══════════════════════════════════════════════════════════════════
#  IMPEDANCE CONTROL — NUCLEO
#
#  τ = Kp·(q_d − q) − Kd·dq + g(q)
#  ctrl_i = q_i + (τ_i − biasprm2_i·dq_i) / gainprm_i
# ══════════════════════════════════════════════════════════════════
def apply_impedance_torque(q_des: np.ndarray,
                           Kp: np.ndarray,
                           Kd: np.ndarray,
                           gravity_comp: bool = True) -> np.ndarray:
    q  = np.array([data.qpos[adr] for adr in all_qposadr])
    dq = np.array([data.qvel[adr] for adr in all_dofadr])

    tau = Kp @ (q_des - q) - Kd @ dq

    if gravity_comp:
        tau += np.array([data.qfrc_bias[adr] for adr in all_dofadr])

    tau = np.clip(tau, -FORCE_LIMITS, FORCE_LIMITS)

    ctrl_arm = q + (tau - arm_biasprm2 * dq) / arm_gainprm
    data.ctrl[ACT_ARM] = ctrl_arm
    return tau


# ══════════════════════════════════════════════════════════════════
#  INIZIALIZZAZIONE CICLO
# ══════════════════════════════════════════════════════════════════
door_index  = 0
cycle_count = 0
setup_door(DOOR_SEQUENCE[door_index])
print(f"\n→ Ciclo 1/{N_CYCLES} | porta: {active['label']}")

q_interp_start      = Q_NEUTRAL.copy()
slitta_interp_start = SLITTA_NEUTRAL

phase           = MOVE_SLITTA_TO_HANDLE
phase_timer     = 0
traj_index      = 1
close_index     = 0
traj_wait_timer = 0
best_dist       = float("inf")
stall_counter   = 0

MAX_STEPS = int(60_000 * N_CYCLES * len(DOOR_SEQUENCE)) + 10_000

# ══════════════════════════════════════════════════════════════════
#  LOOP DI SIMULAZIONE
# ══════════════════════════════════════════════════════════════════
with mujoco.viewer.launch_passive(model, data) as viewer:

    for step in range(MAX_STEPS):

        if phase == DONE:
            break

        mujoco.mj_forward(model, data)

        # ── Selezione profilo impedenza ───────────────────────────
        if phase in CONTACT_PHASES:
            Kp_active, Kd_active = KP_CONTACT, KD_CONTACT
        else:
            Kp_active, Kd_active = KP_FREE, KD_FREE

        # ══════════════════════════════════════════════════════════
        #  STATE MACHINE
        # ══════════════════════════════════════════════════════════
        match phase:

            # ── 0: slitta avanza verso la maniglia ────────────────
            case 0:
                interpolate_arm(q_interp_start, active["q_home"],
                                slitta_interp_start, active["slitta_at_handle"])

                if slitta_reached(active["slitta_at_handle"]):
                    q_des_ik[:]   = data.qpos[ik_qposadr].copy()
                    traj_index    = 1
                    traj_wait_timer = 0
                    reset_integral()
                    best_dist     = float("inf")
                    stall_counter = 0
                    phase         = APPROACH_HANDLE
                    phase_timer   = 0

            # ── 1: IK verso la maniglia ───────────────────────────
            case 1:
                if step % ik_every == 0:
                    target_pos = active["traj_open"][0, 0:3]
                    dist = solve_position_ik(target_pos, gain_approach)

                    if dist < best_dist - 0.001:
                        best_dist     = dist
                        stall_counter = 0
                    else:
                        stall_counter += 1

                    if dist < DIST_APPROACH or stall_counter >= STALL_LIMIT:
                        reset_integral()
                        phase       = CLOSE_GRIPPER
                        phase_timer = 0

            # ── 2: chiudi gripper e attiva weld ───────────────────
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

            # ── 3: stabilizza weld ────────────────────────────────
            case 3:
                global q_hold_open
                data.eq_active[weld_id] = 1
                phase_timer += 1

                if phase_timer >= STABILIZE_STEPS:
                    q_hold_open = q_des_all.copy()
                    reset_integral()
                    phase       = OPEN_DOOR
                    phase_timer = 0

            # ── 4: apertura porta — segue traiettoria .npy ────────
            case 4:
                # apertura porta tramite slitta
                data.eq_active[weld_id] = 1

                # il braccio resta nella configurazione in cui ha afferrato la maniglia
                q_des_all[:] = q_hold_open.copy()
                q_des_ik[:]  = q_hold_open[:len(q_des_ik)].copy()
                lock_fixed_joints()

                if slitta_reached(active["slitta_door_open"]):
                    reset_integral()
                    phase       = WAIT_OPEN
                    phase_timer = 0

            # ── 5: pausa porta aperta ─────────────────────────────
            case 5:
                data.eq_active[weld_id] = 1
                phase_timer += 1
                if phase_timer >= WAIT_OPEN_STEPS:
                    data.qvel[:] = 0.0
                    data.qacc[:] = 0.0
                    mujoco.mj_forward(model, data)
                    close_index     = 0
                    traj_wait_timer = 0
                    reset_integral()
                    phase       = CLOSE_DOOR
                    phase_timer = 0

            # ── 6: chiusura porta — traiettoria inversa ───────────
            case 6:
                # chiusura porta tramite slitta
                data.eq_active[weld_id] = 1

                q_des_all[:] = q_hold_open.copy()
                q_des_ik[:]  = q_hold_open[:len(q_des_ik)].copy()
                lock_fixed_joints()

                if slitta_reached(active["slitta_at_handle"]):
                    reset_integral()
                    phase       = HOLD_CLOSED
                    phase_timer = 0

            # ── 7: tieni chiusa ───────────────────────────────────
            case 7:
                data.eq_active[weld_id] = 1
                phase_timer += 1
                if phase_timer >= HOLD_STEPS:
                    data.eq_active[weld_id] = 0
                    q_interp_start      = q_des_all.copy()
                    slitta_interp_start = float(data.qpos[slitta_qposadr])
                    reset_integral()
                    phase       = RETURN_NEUTRAL
                    phase_timer = 0

            # ── 8: slitta e braccio tornano al neutro ─────────────
            case 8:
                interpolate_arm(q_interp_start, Q_NEUTRAL,
                                slitta_interp_start, SLITTA_NEUTRAL)

                if slitta_reached(SLITTA_NEUTRAL):
                    data.qpos[slitta_qposadr] = SLITTA_NEUTRAL
                    data.qvel[slitta_dofadr]  = 0.0
                    q_des_all[:] = Q_NEUTRAL.copy()
                    q_des_ik[:]  = Q_NEUTRAL[:5].copy()
                    lock_fixed_joints()
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
                        print(f"\n→ Ciclo {cycle_count + 1}/{N_CYCLES} "
                              f"| porta: {door_configs[next_key]['label']}")
                        setup_door(next_key)
                        q_interp_start      = Q_NEUTRAL.copy()
                        slitta_interp_start = SLITTA_NEUTRAL
                        phase           = MOVE_SLITTA_TO_HANDLE
                        phase_timer     = 0
                        traj_index      = 1
                        close_index     = 0
                        traj_wait_timer = 0
                        best_dist       = float("inf")
                        stall_counter   = 0

        # ══════════════════════════════════════════════════════════
        #  SET-POINT SLITTA
        # ══════════════════════════════════════════════════════════
        if phase == MOVE_SLITTA_TO_HANDLE:
            data.ctrl[ACT_SLITTA] = active["slitta_at_handle"]

        elif phase in (APPROACH_HANDLE, CLOSE_GRIPPER, STABILIZE_WELD,
                    WAIT_OPEN, CLOSE_DOOR, HOLD_CLOSED):
            data.ctrl[ACT_SLITTA] = active["slitta_at_handle"]

        elif phase == OPEN_DOOR:
            data.ctrl[ACT_SLITTA] = active["slitta_door_open"]

        elif phase == CLOSE_DOOR:
            data.ctrl[ACT_SLITTA] = active["slitta_at_handle"]

        elif phase == RETURN_NEUTRAL:
            data.ctrl[ACT_SLITTA] = SLITTA_NEUTRAL

        # ══════════════════════════════════════════════════════════
        #  IMPEDANCE CONTROL + GRIPPER
        # ══════════════════════════════════════════════════════════
        if phase != DONE:
            hard_lock_fixed_joints()
            apply_impedance_torque(q_des_all, Kp_active, Kd_active,
                                   gravity_comp=True)

            open_phases = (MOVE_SLITTA_TO_HANDLE, APPROACH_HANDLE, RETURN_NEUTRAL)
            data.ctrl[ACT_GRIPPER] = 255 if phase in open_phases else 50

            mujoco.mj_step(model, data)
            viewer.sync()

        # ── Log ───────────────────────────────────────────────────
        if step % 50 == 0 and phase != DONE:
            t          = step * model.opt.timestep
            door_pos   = data.qpos[door_qposadr]
            slitta_pos = data.qpos[slitta_qposadr]
            q          = np.array([data.qpos[adr] for adr in all_qposadr])
            tau_norm   = float(np.linalg.norm(
                Kp_active @ (q_des_all - q)
                - Kd_active @ np.array([data.qvel[a] for a in all_dofadr])
            ))
            profile  = "CONTACT" if phase in CONTACT_PHASES else "FREE   "
            wp_info  = (f"wp {traj_index}/{active['n_open']-1}"
                        if phase in (OPEN_DOOR, CLOSE_DOOR) else "")
            print(
                f"t={t:6.2f}s | ciclo {cycle_count+1}/{N_CYCLES} "
                f"| porta {active.get('label','?'):>8s} | {STATE_NAMES[phase]}"
                f"| slitta={slitta_pos:+.4f} m | porta={door_pos:+.4f} m "
                f"| |τ|={tau_norm:5.1f} N·m | [{profile}] {wp_info}"
            )

print("Simulazione terminata.")