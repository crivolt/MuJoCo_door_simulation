import mujoco
import mujoco.viewer
import numpy as np
import scipy.io

# LOAD MODEL
model = mujoco.MjModel.from_xml_path("/Users/cristianvoltan/Desktop/unipd/tirocinio/CAD/electrolux/electrolux.xml")
data = mujoco.MjData(model)

# JOINT-SPACE IMPEDANCE CONTROL
# The arm actuators are configured as joint-space impedance controllers:
#
#   tau = Kq * (q_des - q) - Dq * qdot
#
# where:
#   q_des is sent through data.ctrl[arm_actuators]
#   Kq is the joint stiffness   → gainprm[0] and -biasprm[1]
#   Dq is the joint damping     → -biasprm[2]

arm_actuators = [0, 1, 2, 3, 4, 5]

K_APPROACH = 700.0
D_APPROACH = 200.0

K_GRASP   = 800.0
D_GRASP   = 120.0

K_OPENING = 2500.0
D_OPENING = 180.0

K_CLOSING = 2500.0
D_CLOSING = 180.0

K_HOLD = 800.0
D_HOLD = 300.0

JOINT_FORCE_LIMITS = np.array([330, 330, 150, 56, 56, 56])


def set_joint_impedance(Kq, Dq, force_limit):
    """Set joint-space impedance parameters for all arm actuators."""
    for i, a in enumerate(arm_actuators):
        model.actuator_gainprm[a, 0] = Kq
        model.actuator_biasprm[a, 1] = -Kq
        model.actuator_biasprm[a, 2] = -Dq
        model.actuator_forcerange[a, 0] = -force_limit[i]
        model.actuator_forcerange[a, 1] =  force_limit[i]


# Apply initial impedance
set_joint_impedance(K_APPROACH, D_APPROACH, JOINT_FORCE_LIMITS)

# JOINTS
all_joint_names = ["shoulder_pan_joint", "shoulder_lift_joint", "elbow_joint", "wrist_1_joint", "wrist_2_joint", "wrist_3_joint"]

all_qposadr = []  # joint position addresses
all_dofadr  = []  # joint velocity addresses

for name in all_joint_names:
    jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)
    all_qposadr.append(model.jnt_qposadr[jid])
    all_dofadr.append(model.jnt_dofadr[jid])

all_qposadr = np.array(all_qposadr)
all_dofadr  = np.array(all_dofadr)

# DOOR JOINT
door_joint_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "giunto_porta")
door_qposadr  = model.jnt_qposadr[door_joint_id]
door_dofadr   = model.jnt_dofadr[door_joint_id]

model.dof_frictionloss[door_dofadr] = 62.06


# SITES AND WELD CONSTRAINT
site_id        = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, "attachment_site")
target_site_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, "target_maniglia")
weld_id        = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_EQUALITY, "presa")


# HOME POSITION
q_home = np.array([0.0628, -1.88, -0.723, -2.58, 0.0, 0.0])

for i in range(len(all_joint_names)):
    data.qpos[all_qposadr[i]] = q_home[i]

data.ctrl[arm_actuators] = q_home
mujoco.mj_forward(model, data)

q_des_all = data.qpos[all_qposadr].copy()


# TRAJECTORY
trajectory = np.load("/Users/cristianvoltan/Desktop/unipd/tirocinio/CAD/electrolux/traiettorie/traiettoria_apertura.npy")

OPEN_DEG  = 62
T_OPEN    = 1.0
T_CLOSE   = 1.0

n_points_open = int(len(trajectory) * OPEN_DEG / 180)
n_points_open = max(2, min(n_points_open, len(trajectory)))

trajectory_close = trajectory[:n_points_open][::-1].copy()

target_pos = trajectory[0, 0:3]  # starting point for approach phase


# STATE MACHINE
APPROACH_HANDLE = 0
CLOSE_GRIPPER   = 1
STABILIZE_WELD  = 2
OPEN_DOOR       = 3
WAIT_OPEN       = 4
CLOSE_DOOR      = 5
HOLD_CLOSED     = 6
STABILIZE_OPEN  = 7

# SIMULATION PARAMETERS
NUM_CYCLES    = 1
current_cycle = 0

gain_phase0 = 0.001
gain_phase3 = 0.03
gain_close  = 0.02

damping  = 5e-3
ik_every = 1

phase       = APPROACH_HANDLE
phase_timer = 0

CLOSE_DURATION = 500
HOLD_STEPS     = int(15.0 / model.opt.timestep)

traj_index = 1

best_dist     = float("inf")
stall_counter = 0
STALL_LIMIT   = 2000

WAIT_BEFORE_CLOSE_SEC   = 4.0
WAIT_BEFORE_CLOSE_STEPS = int(WAIT_BEFORE_CLOSE_SEC / model.opt.timestep)

# CARTESIAN PI CONTROLLER (used only inside IK to generate q_des)
KP           = 1.20
KI           = 0.10
INTEGRAL_CLIP = 0.05

error_integral = np.zeros(3)


def reset_integral():
    """Reset the Cartesian integral error."""
    global error_integral
    error_integral[:] = 0.0



# POLYNOMIAL PROFILE
# θ(s) = a0 + a1·s + a2·s² + a3·s³,  s ∈ [0, 1]
#
# Boundary conditions:
#   θ(0) = 0        → starts at zero
#   θ(1) = theta_f  → ends at target angle
#   ω(0) = 0        → zero velocity at start
#   ω(1) = 0        → zero velocity at end

def compute_cubic(theta_f=62.0):
    """Solve for cubic polynomial coefficients [a0, a1, a2, a3]."""
    A = np.array([
        [1, 0, 0, 0],   # θ(0) = 0
        [1, 1, 1, 1],   # θ(1) = theta_f
        [0, 1, 0, 0],   # ω(0) = 0
        [0, 1, 2, 3],   # ω(1) = 0
    ])
    b = np.array([0, theta_f, 0, 0])
    return np.linalg.solve(A, b)


def cubic_theta(s, c):
    """Evaluate cubic polynomial at normalized time s."""
    s = np.clip(s, 0.0, 1.0)
    return c[0] + c[1]*s + c[2]*s**2 + c[3]*s**3


# INVERSE KINEMATICS WITH PI CONTROLLER
def solve_position_ik(target, gain):
    """
    Compute desired joint positions via damped least-squares IK.

    The resulting q_des is used as the equilibrium point for the
    joint-space impedance controller.
    """
    global q_des_all, error_integral

    mujoco.mj_forward(model, data)

    site_pos  = data.site_xpos[site_id].copy()
    error_pos = target - site_pos

    # PI with anti-windup
    error_integral += error_pos * model.opt.timestep
    error_integral  = np.clip(error_integral, -INTEGRAL_CLIP, INTEGRAL_CLIP)
    control = KP * error_pos + KI * error_integral

    # Jacobian at the end-effector site
    jacp = np.zeros((3, model.nv))
    mujoco.mj_jacSite(model, data, jacp, None, site_id)
    J = jacp[:, all_dofadr]

    # Damped least-squares solution
    A  = J @ J.T + damping * np.eye(3)
    dq = J.T @ np.linalg.solve(A, control)

    q_des_all += gain * dq

    # Enforce joint limits
    for i, name in enumerate(all_joint_names):
        jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)
        if model.jnt_limited[jid]:
            q_min, q_max = model.jnt_range[jid]
            q_des_all[i] = np.clip(q_des_all[i], q_min, q_max)

    return np.linalg.norm(error_pos)



# LOGGING AND INITIALIZATION
log_time     = []
log_door_deg = []
log_door_vel = []
log_tau      = []

cubic_c          = compute_cubic(theta_f=OPEN_DEG)
open_start_time  = None
close_start_time = None

MAX_STEPS = 30000 * NUM_CYCLES

# SIMULATION LOOP
with mujoco.viewer.launch_passive(model, data) as viewer:

    for step in range(MAX_STEPS):

        match phase:

            
            # PHASE 0: APPROACH HANDLE
            # Move end-effector to the door handle using IK.
            # Exits when the position error is small or the IK stalls.
            case 0:
                set_joint_impedance(K_APPROACH, D_APPROACH, JOINT_FORCE_LIMITS)

                if step % ik_every == 0:
                    dist = solve_position_ik(target_pos, gain_phase0)

                    if dist < best_dist - 0.001:
                        best_dist = dist
                        stall_counter = 0
                    else:
                        stall_counter += 1

                    if dist < 0.005 or stall_counter >= STALL_LIMIT:
                        reset_integral()
                        phase = CLOSE_GRIPPER
                        phase_timer = 0

            # PHASE 1: CLOSE GRIPPER
            # Wait for the gripper to close, then activate the weld constraint.
            case 1:
                set_joint_impedance(K_GRASP, D_GRASP, JOINT_FORCE_LIMITS)

                phase_timer += 1

                if phase_timer >= CLOSE_DURATION:
                    data.qvel[:] = 0.0
                    data.qacc[:] = 0.0
                    mujoco.mj_forward(model, data)

                    data.eq_active[weld_id] = 1
                    mujoco.mj_forward(model, data)

                    data.qvel[:] = 0.0
                    data.qacc[:] = 0.0
                    mujoco.mj_forward(model, data)

                    reset_integral()
                    phase = STABILIZE_WELD
                    phase_timer = 0

            # PHASE 2: STABILIZE WELD
            # Let the system settle after the weld is activated.
            case 2:
                set_joint_impedance(K_GRASP, D_GRASP, JOINT_FORCE_LIMITS)

                data.eq_active[weld_id] = 1
                phase_timer += 1

                if phase_timer >= 500:
                    data.qvel[:] = 0.0
                    data.qacc[:] = 0.0
                    mujoco.mj_forward(model, data)

                    traj_index = 1
                    reset_integral()
                    phase = OPEN_DOOR

            # PHASE 3: OPEN DOOR
            # Follow the opening trajectory using a cubic velocity profile.
            # Friction drops from high to low after 1° to model real behavior.
            case 3:
                set_joint_impedance(K_OPENING, D_OPENING, JOINT_FORCE_LIMITS)
                data.eq_active[weld_id] = 1

                if open_start_time is None:
                    open_start_time = step * model.opt.timestep

                t_open = step * model.opt.timestep - open_start_time

                # Normalized time s ∈ [0, 1]
                s = np.clip(t_open / T_OPEN, 0.0, 1.0)

                # Map s through cubic profile to trajectory index
                theta_norm = cubic_theta(s, cubic_c) / OPEN_DEG
                traj_index = int(theta_norm * (n_points_open - 1))
                traj_index = np.clip(traj_index, 0, n_points_open - 1)

                # Door friction: high until latch releases at ~1°
                door_angle_now = np.rad2deg(data.qpos[door_qposadr])
                if door_angle_now < 1.0:
                    model.dof_frictionloss[door_dofadr] = 62.06
                else:
                    model.dof_frictionloss[door_dofadr] = 6.206

                dist = solve_position_ik(trajectory[traj_index, 0:3], gain_phase3)

                if s >= 1.0:
                    reset_integral()
                    q_des_all[:] = data.qpos[all_qposadr].copy()
                    open_start_time = None
                    phase = WAIT_OPEN
                    phase_timer = 0

            # PHASE 4: WAIT WITH DOOR OPEN
            # Hold position for a fixed time before closing.
            case 4:
                set_joint_impedance(K_HOLD, D_HOLD, JOINT_FORCE_LIMITS)
                data.eq_active[weld_id] = 1

                phase_timer += 1

                if phase_timer >= WAIT_BEFORE_CLOSE_STEPS:
                    data.qvel[:] = 0.0
                    data.qacc[:] = 0.0
                    mujoco.mj_forward(model, data)

                    close_start_time = None
                    reset_integral()
                    phase = CLOSE_DOOR

            # PHASE 5: CLOSE DOOR
            # Follow the closing trajectory (reverse of opening) using the
            # same cubic velocity profile.
            case 5:
                set_joint_impedance(K_CLOSING, D_CLOSING, JOINT_FORCE_LIMITS)
                data.eq_active[weld_id] = 1

                if close_start_time is None:
                    close_start_time = step * model.opt.timestep

                t_close = step * model.opt.timestep - close_start_time
                s = np.clip(t_close / T_CLOSE, 0.0, 1.0)

                theta_norm  = cubic_theta(s, cubic_c) / OPEN_DEG
                close_index = int(theta_norm * (len(trajectory_close) - 1))
                close_index = np.clip(close_index, 0, len(trajectory_close) - 1)

                dist = solve_position_ik(trajectory_close[close_index, 0:3], gain_close)

                if s >= 1.0:
                    reset_integral()
                    q_des_all[:] = data.qpos[all_qposadr].copy()
                    close_start_time = None
                    phase = HOLD_CLOSED
                    phase_timer = 0

            # PHASE 6: HOLD CLOSED
            # Hold the closed position, then either end or start a new cycle.
            case 6:
                # Freeze desired position at entry to minimize holding torques
                if phase_timer == 1:
                    q_des_all[:] = data.qpos[all_qposadr].copy()

                set_joint_impedance(K_HOLD, D_HOLD, JOINT_FORCE_LIMITS)
                data.eq_active[weld_id] = 1

                phase_timer += 1

                if phase_timer >= HOLD_STEPS:
                    current_cycle += 1
                    print(f"=== Cycle {current_cycle}/{NUM_CYCLES} completed ===")

                    if current_cycle >= NUM_CYCLES:
                        break

                    # Reset state for next cycle
                    traj_index       = 1
                    best_dist        = float("inf")
                    stall_counter    = 0
                    open_start_time  = None
                    close_start_time = None
                    model.dof_frictionloss[door_dofadr] = 62.06
                    q_des_all[:] = data.qpos[all_qposadr].copy()
                    reset_integral()
                    phase       = STABILIZE_OPEN
                    phase_timer = 0

            # PHASE 7: STABILIZE BEFORE NEXT OPENING
            # Short settling phase before starting the next opening cycle.
            case 7:
                set_joint_impedance(K_GRASP, D_GRASP, JOINT_FORCE_LIMITS)
                data.eq_active[weld_id] = 1

                phase_timer += 1

                if phase_timer >= 300:
                    q_des_all[:] = data.qpos[all_qposadr].copy()
                    reset_integral()
                    traj_index  = 1
                    phase       = OPEN_DOOR
                    phase_timer = 0

        # SEND CONTROL AND STEP SIMULATION

        data.ctrl[arm_actuators] = q_des_all

        joint_torques = data.qfrc_actuator[all_dofadr]

        mujoco.mj_step(model, data)

        # Log data after step for accurate state recording
        log_time.append(step * model.opt.timestep)
        log_door_deg.append(np.rad2deg(data.qpos[door_qposadr]))
        log_door_vel.append(np.rad2deg(data.qvel[door_dofadr]))
        log_tau.append(joint_torques.copy())

        viewer.sync()

        # Print diagnostics
        if step % 50 == 0:
            door_angle_deg = np.rad2deg(data.qpos[door_qposadr])
            time_sec       = step * model.opt.timestep
            joint_torques  = data.qfrc_actuator[all_dofadr]
            door_torque    = data.qfrc_actuator[door_dofadr]
            door_vel       = np.rad2deg(data.qvel[door_dofadr])

            REAL_LIMITS = np.array([330, 330, 150, 56, 56, 56])
            violations  = np.abs(joint_torques) > REAL_LIMITS

            print(
                f"Time {time_sec:.2f} | phase {phase} | "
                f"door: {door_angle_deg:.2f}° | vel: {door_vel:.1f}°/s | "
                f"tau_door: {door_torque:.2f} | tau: {np.round(joint_torques, 2)} | "
                f"{'*** VIOLATION J' + str(list(np.where(violations)[0] + 1)) + ' ***' if violations.any() else 'ok'}"
            )

# FINAL OUTPUT
door_angle_deg = np.rad2deg(data.qpos[door_qposadr])
print(f"Final door angle: {door_angle_deg:.2f}°")

scipy.io.savemat("dati_porta.mat", {
    "tempo":    np.array(log_time),
    "apertura": np.array(log_door_deg),
    "velocita": np.array(log_door_vel),
    "tau":      np.array(log_tau),
})