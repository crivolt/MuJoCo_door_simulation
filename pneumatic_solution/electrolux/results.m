clear
clc
close all

% Load data
data = load("/Users/cristianvoltan/Desktop/unipd/tirocinio/CAD/pneumatic_solution/electrolux/dati_porta_pneumatica_electrolux.mat");

t = data.tempo(:);

door_angle = data.apertura(:);
door_velocity = data.velocita(:);
cylinder_extension = data.posizione_cilindro(:);
force = data.forza(:);

quat_base = data.quat_snodo_base;

% Cylinder linear velocity [m/s]
cylinder_velocity = gradient(cylinder_extension, t);

% Mechanical actuation power [W]
power = force .* cylinder_velocity;

% Convert base joint quaternion to rotations about X, Y, Z
joint_angles = quat_to_euler_xyz(quat_base);

joint_x = rad2deg(joint_angles(:, 1));
joint_y = rad2deg(joint_angles(:, 2));
joint_z = rad2deg(joint_angles(:, 3));


% Figure 1 - Door opening angle
figure("Name", "Door opening angle")
plot(t, door_angle, "LineWidth", 3.0)
grid on
xlabel("Time [s]", "FontSize", 40)
ylabel("Door angle [deg]", "FontSize", 40)
set(gca, "FontSize", 30)

% Figure 2 - Cylinder extension
figure("Name", "Cylinder extension")
plot(t, cylinder_extension, "LineWidth", 3.0)
grid on
xlabel("Time [s]", "FontSize", 30)
ylabel("Cylinder extension [m]", "FontSize", 30)
title("Cylinder Extension Over Time", "FontSize", 36)
legend("Cylinder", "Location", "best", "FontSize", 30)
set(gca, "FontSize", 25)

% Figure 3 - Door angular velocity
figure("Name", "Door angular velocity")
plot(t, door_velocity, "LineWidth", 3.0)
grid on
xlabel("Time [s]", "FontSize", 40)
ylabel("Door angular velocity [deg/s]", "FontSize", 40)
set(gca, "FontSize", 30)

% Figure 4 - Base joint rotation components
figure("Name", "Base joint rotation components")
plot(t, joint_x, "LineWidth", 3.0)
hold on
plot(t, joint_y, "LineWidth", 3.0)
plot(t, joint_z, "LineWidth", 3.0)
grid on
xlabel("Time [s]", "FontSize", 40)
ylabel("Rotation [deg]", "FontSize", 40)
legend("Rotation about X", "Rotation about Y", "Rotation about Z", "Location", "best", "FontSize", 30)
set(gca, "FontSize", 30)

% Figure 5 - Pneumatic actuator force
figure("Name", "Pneumatic actuator force")
plot(t, force, "LineWidth", 3.0)
grid on
xlabel("Time [s]", "FontSize", 40)
ylabel("Actuator force [N]", "FontSize", 40)
set(gca, "FontSize", 30)

% Figure 6 - Mechanical actuation power
figure("Name", "Mechanical actuation power")
plot(t, power, "LineWidth", 3.0)
grid on
xlabel("Time [s]", "FontSize", 40)
ylabel("Mechanical power [W]", "FontSize", 40)
set(gca, "FontSize", 30)


% Figure 7 - Complete overview
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