clear all
clc
close all

% Load data
data = load("/Users/cristianvoltan/Desktop/unipd/tirocinio/CAD/electric_solution/studio54/dati_porta_elettromeccanica_studio_54.mat");
t = data.tempo(:);
door_pos = data.posizione_porta(:);
door_vel = data.velocita_porta(:);
motor_pos = data.posizione_motoriduttore(:);
comp_radial = data.correzione_allineamento_radiale_mm(:);
align_rot = data.rotazione_allineamento_deg(:);
torque = data.coppia_motoriduttore(:);
power = data.potenza_meccanica(:);

% Figure 1 - Door position
figure("Name", "Door position")
plot(t, door_pos, "LineWidth", 3.0)
grid on
xlabel("Time [s]", "FontSize", 40)
ylabel("Door angle [deg]", "FontSize", 40)
set(gca, "FontSize", 30)

% Figure 2 - Door velocity
figure("Name", "Door velocity")
plot(t, door_vel, "LineWidth", 3.0)
grid on
xlabel("Time [s]", "FontSize", 40)
ylabel("Door angular velocity [deg/s]", "FontSize", 40)
set(gca, "FontSize", 30)

% Figure 3 - Gearmotor position
figure("Name", "Gearmotor position")
plot(t, motor_pos, "LineWidth", 3.0)
grid on
xlabel("Time [s]", "FontSize", 40)
ylabel("Motor angle [deg]", "FontSize", 40)
title("Gearmotor Position Over Time", "FontSize", 36)
legend("Motor", "Location", "best", "FontSize", 30)
set(gca, "FontSize", 25)

% Figure 4 - Radial alignment correction

figure("Name", "Radial alignment correction")

plot(t, comp_radial, "LineWidth", 3.0)

grid on

xlabel("Time [s]", "FontSize", 40)

ylabel("Radial correction [mm]", "FontSize", 40)

set(gca, "FontSize", 30)

% Figure 5 - Alignment joint rotation

figure("Name", "Alignment joint rotation")

plot(t, align_rot, "LineWidth", 3.0)

grid on

xlabel("Time [s]", "FontSize", 40)

ylabel("Rotation [deg]", "FontSize", 40)

set(gca, "FontSize", 30)

% Figure 5 - Complete overview
figure("Name", "Complete overview")
tiledlayout(5, 1)

nexttile
plot(t, door_pos, "LineWidth", 1.5)
grid on
ylabel("Angle [deg]")
title("Door Position")
legend("Door", "Location", "best")

nexttile
plot(t, door_vel, "LineWidth", 1.5)
grid on
ylabel("Velocity [deg/s]")
title("Door Velocity")
legend("Door", "Location", "best")

nexttile
plot(t, motor_pos, "LineWidth", 1.5)
grid on
ylabel("Angle [deg]")
title("Gearmotor Position")
legend("Motor", "Location", "best")

nexttile
plot(t, comp_radial, "LineWidth", 1.5)
grid on
ylabel("Correction [mm]")
title("Radial Alignment Correction")
legend("Radial correction", "Location", "best")

nexttile
plot(t, align_rot, "LineWidth", 1.5)
grid on
xlabel("Time [s]")
ylabel("Rotation [deg]")
title("Alignment Joint Rotation")
legend("Rotation", "Location", "best")

% Figure 6 - Motor actuation torque

figure("Name", "Motor actuation torque")

plot(t, torque, "LineWidth", 3.0)

grid on

xlabel("Time [s]", "FontSize", 40)

ylabel("Motor torque [Nm]", "FontSize", 40)

set(gca, "FontSize", 30)

% Figure 7 - Mechanical actuation power

figure("Name", "Mechanical actuation power")

plot(t, power, "LineWidth", 3.0)

grid on

xlabel("Time [s]", "FontSize", 40)

ylabel("Mechanical power [W]", "FontSize", 40)

set(gca, "FontSize", 30)