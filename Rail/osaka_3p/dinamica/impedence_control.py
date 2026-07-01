import mujoco
import mujoco.viewer
import numpy as np

# ══════════════════════════════════════════════════════════════════
#  PARAMETRI PRINCIPALI  ← modifica solo qui
# ══════════════════════════════════════════════════════════════════
N_CYCLES = 2
OPEN_DEG = 62

XML_PATH     = "/Users/cristianvoltan/Desktop/unipd/tirocinio/CAD/Rail/osaka_3p/Osaka_3p_finale.xml"
TRAJ_SX_PATH = "/Users/cristianvoltan/Desktop/unipd/tirocinio/CAD/Rail/osaka_3p/traiettorie/traiettoria_sinistra.npy"
TRAJ_DX_PATH = "/Users/cristianvoltan/Desktop/unipd/tirocinio/CAD/Rail/osaka_3p/traiettorie/traiettoria_destra.npy"

# ══════════════════════════════════════════════════════════════════
#  PARAMETRI IMPEDANCE CONTROL
# ══════════════════════════════════════════════════════════════════
#
#  τ = M(q) · [ Kp·(q_des − q) + Kd·(q̇_des − q̇) ] + bias(q, q̇)
#
#  bias = C(q,q̇)·q̇ + g(q)  →  già in data.qfrc_bias di MuJoCo
#
#  Kp : rigidità virtuale   [N·m/rad]
#  Kd : smorzamento virtuale [N·m·s/rad]
#       regola critico: Kd ≈ 2·sqrt(Kp)  (per inerzia ≈ 1 kg·m²)
#
IMP_KP = np.array([600.0, 600.0, 500.0, 500.0, 1000.0, 1000.0, 600.0])
IMP_KD = np.array([ 50.0,  50.0,  45.0,  45.0,  80.0,  80.0,  60.0])

# Limiti forza per sicurezza [N·m]
IMP_FORCE_LIMIT = 150.0

# ── Slitta (rimane PD classico, non serve impedance su prismatico) ─
SLITTA_KP = 3000.0
SLITTA_KD = 2500.0
# ══════════════════════════════════════════════════════════════════

model = mujoco.MjModel.from_xml_path(XML_PATH)
data  = mujoco.MjData(model)

# ── Indici attuatori ──────────────────────────────────────────────
ACT_SLITTA  = 0
ACT_ARM     = [1, 2, 3, 4, 5, 6, 7]
ACT_GRIPPER = 8

# ══════════════════════════════════════════════════════════════════
#  CONFIGURAZIONE ATTUATORI IN MODALITÀ COPPIA PURA
#  gainprm[0] = 1  →  ctrl interpretato direttamente come τ [N·m]
#  biastype   = 0  →  nessun bias interno (lo calcoliamo noi)
# ══════════════════════════════════════════════════════════════════
for a in ACT_ARM:
    model.actuator_gainprm[a, 0]  =  1.0
    model.actuator_biastype[a]    =  0
    model.actuator_biasprm[a, :]  =  0.0
    # CRITICO: ctrlrange deve essere in N*m, non in radianti!
    # Il ctrlrange originale (es. +-2.89 rad) clampava ctrl a ~3 N*m -> robot cade
    model.actuator_ctrlrange[a, 0]  = -IMP_FORCE_LIMIT
    model.actuator_ctrlrange[a, 1]  =  IMP_FORCE_LIMIT
    model.actuator_forcerange[a, 0] = -IMP_FORCE_LIMIT
    model.actuator_forcerange[a, 1] =  IMP_FORCE_LIMIT

# ── Joint del braccio ─────────────────────────────────────────────
all_joint_names = ["joint1","joint2","joint3","joint4","joint5","joint6","joint7"]
ik_joint_names  = all_joint_names  # stesso set

all_qposadr, all_dofadr = [], []
for name in all_joint_names:
    jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)
    all_qposadr.append(model.jnt_qposadr[jid])
    all_dofadr.append(model.jnt_dofadr[jid])
all_qposadr = np.array(all_qposadr)
all_dofadr  = np.array(all_dofadr)

ik_qposadr = all_qposadr
ik_dofadr  = all_dofadr

# ── Slitta ────────────────────────────────────────────────────────
slitta_joint_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "giunto8_slitta")
slitta_qposadr  = model.jnt_qposadr[slitta_joint_id]
slitta_dofadr   = model.jnt_dofadr[slitta_joint_id]

SLITTA_NEUTRAL = 0.0
SLITTA_POS_SX  = -0.6
SLITTA_POS_DX  =  0.6

# ── Posizione neutra braccio ──────────────────────────────────────
Q_NEUTRAL = np.array([0.0, -1.0, 0.0, -2.16, 0.0, 2.66, 0.811])

# ── Traiettorie ───────────────────────────────────────────────────
def build_traj_close(traj, open_deg, max_deg=95):
    n = max(2, min(int(len(traj) * open_deg / max_deg), len(traj)))
    return traj[:n][::-1].copy(), n

trajectory_sx = np.load(TRAJ_SX_PATH)
trajectory_dx = np.load(TRAJ_DX_PATH)
traj_close_sx, n_open_sx = build_traj_close(trajectory_sx, OPEN_DEG)
traj_close_dx, n_open_dx = build_traj_close(trajectory_dx, OPEN_DEG)

# ── Configurazioni per porta ──────────────────────────────────────
door_configs = {
    "sx": dict(
        label         = "SINISTRA",
        slitta_target = SLITTA_POS_SX,
        q_home        = np.array([-0.956, -0.088, 0.0, -1.36, 0.0, 2.3, 0.881]),
        site_name     = "grasp_site",
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
        q_home        = np.array([ 0.956, -0.088, 0.0, -1.36, 0.0, 2.3, 0.881]),
        site_name     = "grasp_site",
        target_site   = "target_manigliadx",
        weld_name     = "presa_dx",
        door_joint    = "giunto_porta_dx",
        trajectory    = trajectory_dx,
        traj_close    = traj_close_dx,
        n_open        = n_open_dx,
    ),
}

DOOR_SEQUENCE = ["sx", "dx"]

# ── State machine ─────────────────────────────────────────────────
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
    0:"MOVE_TO_DOOR", 1:"APPROACH",   2:"CLOSE_GRIP",
    3:"STABILIZE",    4:"OPEN_DOOR",  5:"WAIT_OPEN",
    6:"CLOSE_DOOR",   7:"HOLD",       8:"RETURN",    9:"DONE"
}

# ── Parametri IK ─────────────────────────────────────────────────
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

# ── Variabili globali runtime ─────────────────────────────────────
error_integral = np.zeros(3)
q_des_all      = np.zeros(len(all_joint_names))
q_des_ik       = np.zeros(len(ik_joint_names))

site_id = target_site_id = weld_id = door_qposadr = None
active  = {}

SLITTA_LOCKED   = False
SLITTA_LOCK_POS = None

q_interp_start      = Q_NEUTRAL.copy()
slitta_interp_start = SLITTA_NEUTRAL

# Buffer inerzia 2-D (nv x nv), riusato ogni step per evitare allocazioni
_M_full = np.zeros((model.nv, model.nv))


# ══════════════════════════════════════════════════════════════════
#  IMPEDANCE CONTROLLER  (cuore del controllo)
# ══════════════════════════════════════════════════════════════════
def compute_impedance_torques(q_des, dq_des=None):
    """
    Calcola le coppie di impedenza nello spazio dei giunti.

    τ = M(q) · [ Kp·(q_des − q) + Kd·(q̇_des − q̇) ]  +  bias(q, q̇)

    Parametri
    ----------
    q_des  : array (7,)   configurazione desiderata
    dq_des : array (7,) | None   velocità desiderata (default 0)

    Ritorna
    -------
    tau : array (7,)   coppia da applicare [N·m]
    """
    if dq_des is None:
        dq_des = np.zeros(len(all_joint_names))

    # Stato corrente
    q_cur  = data.qpos[all_qposadr]          # (7,)
    dq_cur = data.qvel[all_dofadr]            # (7,)

    # Errori
    q_err  = q_des  - q_cur                   # (7,)
    dq_err = dq_des - dq_cur                  # (7,)

    # ── Matrice di inerzia M(q) ridotta ai DOF del braccio ────────
    mujoco.mj_fullM(model, _M_full, data.qM)
    M = _M_full[np.ix_(all_dofadr, all_dofadr)]   # (7x7)

    # ── Termine bias: Coriolis + gravità ──────────────────────────
    #    data.qfrc_bias[dof] = C(q,q̇)·q̇ + g(q)  già calcolato da mj_forward
    bias = data.qfrc_bias[all_dofadr]             # (7,)

    # ── Legge di controllo ────────────────────────────────────────
    tau = M @ (IMP_KP * q_err + IMP_KD * dq_err) + bias

    # Saturazione di sicurezza
    tau = np.clip(tau, -IMP_FORCE_LIMIT, IMP_FORCE_LIMIT)

    return tau


# ══════════════════════════════════════════════════════════════════
#  FUNZIONI DI SUPPORTO
# ══════════════════════════════════════════════════════════════════
def reset_integral():
    global error_integral
    error_integral[:] = 0.0


def setup_door(key):
    global active, q_des_all, q_des_ik, error_integral
    global site_id, target_site_id, weld_id, door_qposadr

    cfg    = door_configs[key]
    active = cfg

    site_id        = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE,    cfg["site_name"])
    target_site_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE,    cfg["target_site"])
    weld_id        = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_EQUALITY, cfg["weld_name"])
    djid           = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT,    cfg["door_joint"])
    door_qposadr   = model.jnt_qposadr[djid]
    error_integral = np.zeros(3)


def interpolate_arm(q_start, q_end, slitta_start, slitta_target):
    """Interpola q_des in base al progresso fisico della slitta."""
    slitta_pos   = data.qpos[slitta_qposadr]
    total_travel = abs(slitta_target - slitta_start)
    traveled     = abs(slitta_pos - slitta_start)
    alpha        = np.clip(traveled / total_travel, 0.0, 1.0) if total_travel > 1e-6 else 1.0
    q_des_all[:] = (1.0 - alpha) * q_start + alpha * q_end
    return alpha


def solve_position_ik(target, gain):
    """IK differenziale con pseudoinversa smorzata (DLS) + termine PI."""
    global q_des_ik, error_integral

    mujoco.mj_forward(model, data)
    site_pos  = data.site_xpos[site_id].copy()
    error_pos = target - site_pos

    error_integral += error_pos * model.opt.timestep
    error_integral  = np.clip(error_integral, -INTEGRAL_CLIP, INTEGRAL_CLIP)

    control = KP_IK * error_pos + KI_IK * error_integral

    jacp = np.zeros((3, model.nv))
    mujoco.mj_jacSite(model, data, jacp, None, site_id)
    J  = jacp[:, ik_dofadr]
    A  = J @ J.T + damping * np.eye(3)
    dq = J.T @ np.linalg.solve(A, control)

    q_des_ik += gain * dq
    for i, name in enumerate(ik_joint_names):
        jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)
        if model.jnt_limited[jid]:
            lo, hi      = model.jnt_range[jid]
            q_des_ik[i] = np.clip(q_des_ik[i], lo, hi)

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

# Inizializza q_des con la posizione corrente (nessun salto iniziale)
q_des_all[:] = Q_NEUTRAL.copy()
q_des_ik[:]  = Q_NEUTRAL.copy()

# Applica subito le coppie di impedenza per stabilizzare dalla pos iniziale
tau0 = compute_impedance_torques(Q_NEUTRAL)
for i, a in enumerate(ACT_ARM):
    data.ctrl[a] = tau0[i]

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

        # ── Forward kinematics (aggiorna site_xpos, qfrc_bias, M) ─
        mujoco.mj_forward(model, data)

        match phase:

            # ── 0 · neutrale → posizione porta ───────────────────
            case 0:
                slitta_tgt = active["slitta_target"]
                slitta_vel = data.qvel[slitta_dofadr]
                slitta_pos = data.qpos[slitta_qposadr]

                alpha = interpolate_arm(q_interp_start, active["q_home"],
                                        slitta_interp_start, slitta_tgt)
                data.ctrl[ACT_SLITTA] = slitta_tgt

                if abs(slitta_pos - slitta_tgt) < 0.01 and abs(slitta_vel) < 0.01:
                    SLITTA_LOCKED   = True
                    SLITTA_LOCK_POS = slitta_tgt
                    data.qpos[slitta_qposadr] = SLITTA_LOCK_POS
                    data.qvel[slitta_dofadr]  = 0.0
                    q_des_all[:] = active["q_home"].copy()
                    q_des_ik[:]  = active["q_home"].copy()
                    mujoco.mj_forward(model, data)
                    reset_integral()
                    phase         = APPROACH_HANDLE
                    phase_timer   = 0
                    best_dist     = float("inf")
                    stall_counter = 0

            # ── 1 · avvicinamento maniglia ────────────────────────
            case 1:
                if step % ik_every == 0:
                    target_pos = active["trajectory"][0, 0:3]
                    dist = solve_position_ik(target_pos, gain_phase0)
                    q_des_all[:] = q_des_ik

                    if dist < best_dist - 0.001:
                        best_dist = dist; stall_counter = 0
                    else:
                        stall_counter += 1

                    if dist < 0.035 or stall_counter >= STALL_LIMIT:
                        reset_integral()
                        phase       = CLOSE_GRIPPER
                        phase_timer = 0

            # ── 2 · chiudi gripper ────────────────────────────────
            case 2:
                phase_timer += 1
                if phase_timer >= CLOSE_DURATION:
                    data.qvel[:] = 0; data.qacc[:] = 0
                    data.eq_active[weld_id] = 1
                    mujoco.mj_forward(model, data)
                    reset_integral()
                    phase       = STABILIZE_WELD
                    phase_timer = 0

            # ── 3 · stabilizza weld ───────────────────────────────
            case 3:
                phase_timer += 1
                data.eq_active[weld_id] = 1
                if phase_timer >= 100:
                    traj_index = 1; traj_wait_timer = 0
                    reset_integral()
                    phase = OPEN_DOOR

            # ── 4 · apri porta ────────────────────────────────────
            case 4:
                if step % ik_every == 0:
                    data.eq_active[weld_id] = 1
                    tgt  = active["trajectory"][traj_index, 0:3]
                    dist = solve_position_ik(tgt, gain_phase3)
                    q_des_all[:] = q_des_ik

                    if dist < 0.03:
                        traj_wait_timer += 1
                        if traj_wait_timer >= TRAJ_WAIT and traj_index < active["n_open"] - 1:
                            traj_index += 1; traj_wait_timer = 0; reset_integral()
                    else:
                        traj_wait_timer = 0

                    if traj_index >= active["n_open"] - 1 and dist < 0.02:
                        reset_integral()
                        phase = WAIT_OPEN; phase_timer = 0

            # ── 5 · attendi a porta aperta ────────────────────────
            case 5:
                data.eq_active[weld_id] = 1
                phase_timer += 1
                if phase_timer >= WAIT_BEFORE_CLOSE_STEPS:
                    data.qvel[:] = 0; data.qacc[:] = 0
                    mujoco.mj_forward(model, data)
                    close_index = 0; traj_wait_timer = 0
                    reset_integral()
                    phase = CLOSE_DOOR

            # ── 6 · chiudi porta ──────────────────────────────────
            case 6:
                if step % ik_every == 0:
                    data.eq_active[weld_id] = 1
                    tgt  = active["traj_close"][close_index, 0:3]
                    dist = solve_position_ik(tgt, gain_close)
                    q_des_all[:] = q_des_ik

                    if dist < 0.03:
                        traj_wait_timer += 1
                        if traj_wait_timer >= TRAJ_WAIT and close_index < len(active["traj_close"]) - 1:
                            close_index += 1; traj_wait_timer = 0; reset_integral()
                    else:
                        traj_wait_timer = 0

                    if close_index >= len(active["traj_close"]) - 1 and dist < 0.002:
                        reset_integral()
                        phase = HOLD_CLOSED; phase_timer = 0

            # ── 7 · tieni chiusa ──────────────────────────────────
            case 7:
                data.eq_active[weld_id] = 1
                phase_timer += 1
                if phase_timer >= int(3.0 / model.opt.timestep):
                    data.eq_active[weld_id] = 0
                    SLITTA_LOCKED = False
                    q_interp_start      = q_des_all.copy()
                    slitta_interp_start = float(data.qpos[slitta_qposadr])
                    reset_integral()
                    phase       = RETURN_NEUTRAL
                    phase_timer = 0

            # ── 8 · posizione porta → neutrale ────────────────────
            case 8:
                slitta_vel = data.qvel[slitta_dofadr]
                slitta_pos = data.qpos[slitta_qposadr]

                alpha = interpolate_arm(q_interp_start, Q_NEUTRAL,
                                        slitta_interp_start, SLITTA_NEUTRAL)
                data.ctrl[ACT_SLITTA] = SLITTA_NEUTRAL

                if abs(slitta_pos - SLITTA_NEUTRAL) < 0.01 and abs(slitta_vel) < 0.01:
                    data.qpos[slitta_qposadr] = SLITTA_NEUTRAL
                    data.qvel[slitta_dofadr]  = 0.0
                    q_des_all[:] = Q_NEUTRAL.copy()
                    q_des_ik[:]  = Q_NEUTRAL.copy()
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
                        print(f"\n→ Ciclo {cycle_count + 1}/{N_CYCLES} | "
                              f"porta: {door_configs[next_key]['label']}")
                        setup_door(next_key)
                        q_interp_start      = Q_NEUTRAL.copy()
                        slitta_interp_start = SLITTA_NEUTRAL
                        phase         = MOVE_TO_DOOR
                        phase_timer   = 0
                        traj_index    = 1
                        close_index   = 0
                        traj_wait_timer = 0
                        best_dist     = float("inf")
                        stall_counter = 0

        # ══════════════════════════════════════════════════════════
        #  APPLICAZIONE COPPIE DI IMPEDENZA  (ogni step)
        # ══════════════════════════════════════════════════════════
        if phase != DONE:

            # Blocco slitta se necessario
            if SLITTA_LOCKED:
                data.qpos[slitta_qposadr] = SLITTA_LOCK_POS
                data.qvel[slitta_dofadr]  = 0.0
                data.ctrl[ACT_SLITTA]     = SLITTA_LOCK_POS

            # ── Impedance torques ─────────────────────────────────
            tau = compute_impedance_torques(q_des_all)
            for i, a in enumerate(ACT_ARM):
                data.ctrl[a] = tau[i]

            # Gripper
            open_gripper_phases = (MOVE_TO_DOOR, APPROACH_HANDLE, RETURN_NEUTRAL)
            data.ctrl[ACT_GRIPPER] = 255 if phase in open_gripper_phases else 50

            mujoco.mj_step(model, data)
            viewer.sync()

            # ── Log ogni 50 step ──────────────────────────────────
            if step % 50 == 0:
                door_angle = np.rad2deg(data.qpos[door_qposadr])
                t          = step * model.opt.timestep
                lbl        = active.get("label", "?")
                spos       = data.qpos[slitta_qposadr]
                tau_norm   = np.linalg.norm(tau)
                print(f"t={t:6.2f}s | ciclo {cycle_count+1}/{N_CYCLES} "
                      f"| porta {lbl:>8s} | {STATE_NAMES[phase]:>12s} "
                      f"| slitta={spos:+.3f} | porta={door_angle:6.1f}° "
                      f"| |τ|={tau_norm:6.1f} N·m")

print("Simulazione terminata.")