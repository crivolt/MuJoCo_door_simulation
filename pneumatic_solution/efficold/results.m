clear
clc
close all

% Load data
data = load("/Users/cristianvoltan/Desktop/unipd/tirocinio/CAD/pneumatic_solution/efficold/dati_porta_pneumatica_efficold.mat");

t = data.tempo(:);

door_angle = data.apertura(:);
door_velocity = data.velocita(:);
cylinder_extension = data.posizione_cilindro(:);

quat_base = data.quat_snodo_base;

% Convert base joint quaternion to rotations about X, Y, Z
joint_angles = quat_to_euler_xyz(quat_base);

joint_x = rad2deg(joint_angles(:, 1));
joint_y = rad2deg(joint_angles(:, 2));
joint_z = rad2deg(joint_angles(:, 3));


% Figure 1 - Door opening angle
figure("Name", "Door opening angle")
plot(t, door_angle, "LineWidth", 1.5)
grid on
xlabel("Time [s]")
ylabel("Door angle [deg]")
title("Door Opening Angle Over Time")
legend("Door", "Location", "best")


% Figure 2 - Cylinder extension
figure("Name", "Cylinder extension")
plot(t, cylinder_extension, "LineWidth", 1.5)
grid on
xlabel("Time [s]")
ylabel("Cylinder extension [m]")
title("Cylinder Extension Over Time")
legend("Cylinder", "Location", "best")


% Figure 3 - Door angular velocity
figure("Name", "Door angular velocity")
plot(t, door_velocity, "LineWidth", 1.5)
grid on
xlabel("Time [s]")
ylabel("Door angular velocity [deg/s]")
title("Door Angular Velocity Over Time")
legend("Door", "Location", "best")


% Figure 4 - Base joint rotation components
figure("Name", "Base joint rotation components")
plot(t, joint_x, "LineWidth", 1.5)
hold on
plot(t, joint_y, "LineWidth", 1.5)
plot(t, joint_z, "LineWidth", 1.5)
grid on
xlabel("Time [s]")
ylabel("Rotation [deg]")
title("Base Joint Rotation Components")
legend("Rotation about X", "Rotation about Y", "Rotation about Z", "Location", "best")


% Figure 5 - Complete overview
figure("Name", "Complete overview")
tiledlayout(4, 1)

nexttile
plot(t, door_angle, "LineWidth", 1.5)
grid on
ylabel("Angle [deg]")
title("Door Opening Angle")
legend("Door", "Location", "best")

nexttile
plot(t, cylinder_extension, "LineWidth", 1.5)
grid on
ylabel("Extension [m]")
title("Cylinder Extension")
legend("Cylinder", "Location", "best")

nexttile
plot(t, door_velocity, "LineWidth", 1.5)
grid on
ylabel("Velocity [deg/s]")
title("Door Angular Velocity")
legend("Door", "Location", "best")

nexttile
plot(t, joint_x, "LineWidth", 1.5)
hold on
plot(t, joint_y, "LineWidth", 1.5)
plot(t, joint_z, "LineWidth", 1.5)
grid on
xlabel("Time [s]")
ylabel("Rotation [deg]")
title("Base Joint Rotation Components")
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