clear
clc
close all

% Load data
data = load("/Users/cristianvoltan/Desktop/unipd/tirocinio/CAD/pneumatic_solution/athen_xl/dati_porta_pneumatica_athen_xl.mat");

t = data.tempo(:);

door_sx = data.posizione_porta_sx(:);
door_dx = data.posizione_porta_dx(:);

vel_sx = data.velocita_porta_sx(:);
vel_dx = data.velocita_porta_dx(:);

cyl_sx = data.posizione_cilindro_sx(:);
cyl_dx = data.posizione_cilindro_dx(:);

quat_sx = data.quat_snodo_stelo_staffa_sx;
quat_dx = data.quat_snodo_stelo_staffa_dx;

% Convert joint quaternion to rotations about X, Y, Z
joint_angles_sx = quat_to_euler_xyz(quat_sx);
joint_angles_dx = quat_to_euler_xyz(quat_dx);

joint_x_sx = rad2deg(joint_angles_sx(:, 1));
joint_y_sx = rad2deg(joint_angles_sx(:, 2));
joint_z_sx = rad2deg(joint_angles_sx(:, 3));

joint_x_dx = rad2deg(joint_angles_dx(:, 1));
joint_y_dx = rad2deg(joint_angles_dx(:, 2));
joint_z_dx = rad2deg(joint_angles_dx(:, 3));


% Figure 1 - Door position
figure("Name", "Door position")
plot(t, door_sx, "LineWidth", 1.5)
hold on
plot(t, door_dx, "LineWidth", 1.5)
grid on
xlabel("Time [s]")
ylabel("Door position [m]")
title("Door Position Over Time")
legend("Left door", "Right door", "Location", "best")


% Figure 2 - Cylinder extension
figure("Name", "Cylinder extension")
plot(t, cyl_sx, "LineWidth", 1.5)
hold on
plot(t, cyl_dx, "LineWidth", 1.5)
grid on
xlabel("Time [s]")
ylabel("Cylinder extension [m]")
title("Cylinder Extension Over Time")
legend("Left cylinder", "Right cylinder", "Location", "best")


% Figure 3 - Door velocity
figure("Name", "Door velocity")
plot(t, vel_sx, "LineWidth", 1.5)
hold on
plot(t, vel_dx, "LineWidth", 1.5)
grid on
xlabel("Time [s]")
ylabel("Door velocity [m/s]")
title("Door Velocity Over Time")
legend("Left door", "Right door", "Location", "best")


% Figure 4 - Left joint rotation components
figure("Name", "Left joint rotation components")
plot(t, joint_x_sx, "LineWidth", 1.5)
hold on
plot(t, joint_y_sx, "LineWidth", 1.5)
plot(t, joint_z_sx, "LineWidth", 1.5)
grid on
xlabel("Time [s]")
ylabel("Rotation [deg]")
title("Left Joint Rotation Components")
legend("Rotation about X", "Rotation about Y", "Rotation about Z", "Location", "best")


% Figure 5 - Right joint rotation components
figure("Name", "Right joint rotation components")
plot(t, joint_x_dx, "LineWidth", 1.5)
hold on
plot(t, joint_y_dx, "LineWidth", 1.5)
plot(t, joint_z_dx, "LineWidth", 1.5)
grid on
xlabel("Time [s]")
ylabel("Rotation [deg]")
title("Right Joint Rotation Components")
legend("Rotation about X", "Rotation about Y", "Rotation about Z", "Location", "best")


% Figure 6 - Complete overview
figure("Name", "Complete overview")
tiledlayout(5, 1)

nexttile
plot(t, door_sx, "LineWidth", 1.5)
hold on
plot(t, door_dx, "LineWidth", 1.5)
grid on
ylabel("Position [m]")
title("Door Position")
legend("Left door", "Right door", "Location", "best")

nexttile
plot(t, cyl_sx, "LineWidth", 1.5)
hold on
plot(t, cyl_dx, "LineWidth", 1.5)
grid on
ylabel("Extension [m]")
title("Cylinder Extension")
legend("Left cylinder", "Right cylinder", "Location", "best")

nexttile
plot(t, vel_sx, "LineWidth", 1.5)
hold on
plot(t, vel_dx, "LineWidth", 1.5)
grid on
ylabel("Velocity [m/s]")
title("Door Velocity")
legend("Left door", "Right door", "Location", "best")

nexttile
plot(t, joint_x_sx, "LineWidth", 1.5)
hold on
plot(t, joint_y_sx, "LineWidth", 1.5)
plot(t, joint_z_sx, "LineWidth", 1.5)
grid on
ylabel("Rotation [deg]")
title("Left Joint Rotation Components")
legend("Rotation about X", "Rotation about Y", "Rotation about Z", "Location", "best")

nexttile
plot(t, joint_x_dx, "LineWidth", 1.5)
hold on
plot(t, joint_y_dx, "LineWidth", 1.5)
plot(t, joint_z_dx, "LineWidth", 1.5)
grid on
xlabel("Time [s]")
ylabel("Rotation [deg]")
title("Right Joint Rotation Components")
legend("Rotation about X", "Rotation about Y", "Rotation about Z", "Location", "best")


% Functions
function eul = quat_to_euler_xyz(q)
    q = double(q);
    q = normalize_quat(q);

    w = q(:, 1);
    x = q(:, 2);
    y = q(:, 3);
    z = q(:, 4);

    rot_x = atan2(2 .* (w .* x + y .* z), 1 - 2 .* (x.^2 + y.^2));

    arg_y = 2 .* (w .* y - z .* x);
    arg_y = max(min(arg_y, 1), -1);
    rot_y = asin(arg_y);

    rot_z = atan2(2 .* (w .* z + x .* y), 1 - 2 .* (y.^2 + z.^2));

    eul = [rot_x rot_y rot_z];
end

function q = normalize_quat(q)
    n = sqrt(sum(q.^2, 2));
    q = q ./ n;
end