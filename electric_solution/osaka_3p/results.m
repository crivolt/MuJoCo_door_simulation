clear
clc
close all

% Load data
data = load("/Users/cristianvoltan/Desktop/unipd/tirocinio/CAD/electric_solution/osaka_3p/dati_porta_meccanica_astana_p.mat");

t = data.tempo(:);

% Porta sinistra
door_pos_sx = data.posizione_porta_sx(:);
door_vel_sx = data.velocita_porta_sx(:);
motor_pos_sx = data.posizione_motoriduttore_sx(:);
comp_radial_sx = data.correzione_allineamento_radiale_sx_mm(:);
align_rot_sx = data.rotazione_allineamento_sx_deg(:);

% Porta destra
door_pos_dx = data.posizione_porta_dx(:);
door_vel_dx = data.velocita_porta_dx(:);
motor_pos_dx = data.posizione_motoriduttore_dx(:);
comp_radial_dx = data.correzione_allineamento_radiale_dx_mm(:);
align_rot_dx = data.rotazione_allineamento_dx_deg(:);


% Figure 1 - Door position
figure("Name", "Door position")
plot(t, door_pos_sx, "LineWidth", 1.5)
hold on
plot(t, door_pos_dx, "LineWidth", 1.5)
grid on
xlabel("Time [s]")
ylabel("Door angle [deg]")
title("Door Position Over Time")
legend("Door SX", "Door DX", "Location", "best")


% Figure 2 - Door velocity
figure("Name", "Door velocity")
plot(t, door_vel_sx, "LineWidth", 1.5)
hold on
plot(t, door_vel_dx, "LineWidth", 1.5)
grid on
xlabel("Time [s]")
ylabel("Door velocity [deg/s]")
title("Door Velocity Over Time")
legend("Door SX", "Door DX", "Location", "best")


% Figure 3 - Gearmotor position
figure("Name", "Gearmotor position")
plot(t, motor_pos_sx, "LineWidth", 1.5)
hold on
plot(t, motor_pos_dx, "LineWidth", 1.5)
grid on
xlabel("Time [s]")
ylabel("Motor angle [deg]")
title("Gearmotor Position Over Time")
legend("Motor SX", "Motor DX", "Location", "best")


% Figure 4 - Radial alignment correction
figure("Name", "Radial alignment correction")
plot(t, comp_radial_sx, "LineWidth", 1.5)
hold on
plot(t, comp_radial_dx, "LineWidth", 1.5)
grid on
xlabel("Time [s]")
ylabel("Radial correction [mm]")
title("Radial Alignment Correction Over Time")
legend("Radial correction SX", "Radial correction DX", "Location", "best")


% Figure 5 - Alignment joint rotation
figure("Name", "Alignment joint rotation")
plot(t, align_rot_sx, "LineWidth", 1.5)
hold on
plot(t, align_rot_dx, "LineWidth", 1.5)
grid on
xlabel("Time [s]")
ylabel("Rotation [deg]")
title("Alignment Joint Rotation Over Time")
legend("Alignment SX", "Alignment DX", "Location", "best")


% Figure 6 - Complete overview
figure("Name", "Complete overview")
tiledlayout(5, 1)

nexttile
plot(t, door_pos_sx, "LineWidth", 1.5)
hold on
plot(t, door_pos_dx, "LineWidth", 1.5)
grid on
ylabel("Angle [deg]")
title("Door Position")
legend("Door SX", "Door DX", "Location", "best")

nexttile
plot(t, door_vel_sx, "LineWidth", 1.5)
hold on
plot(t, door_vel_dx, "LineWidth", 1.5)
grid on
ylabel("Velocity [deg/s]")
title("Door Velocity")
legend("Door SX", "Door DX", "Location", "best")

nexttile
plot(t, motor_pos_sx, "LineWidth", 1.5)
hold on
plot(t, motor_pos_dx, "LineWidth", 1.5)
grid on
ylabel("Angle [deg]")
title("Gearmotor Position")
legend("Motor SX", "Motor DX", "Location", "best")

nexttile
plot(t, comp_radial_sx, "LineWidth", 1.5)
hold on
plot(t, comp_radial_dx, "LineWidth", 1.5)
grid on
ylabel("Correction [mm]")
title("Radial Alignment Correction")
legend("Radial SX", "Radial DX", "Location", "best")

nexttile
plot(t, align_rot_sx, "LineWidth", 1.5)
hold on
plot(t, align_rot_dx, "LineWidth", 1.5)
grid on
xlabel("Time [s]")
ylabel("Rotation [deg]")
title("Alignment Joint Rotation")
legend("Rotation SX", "Rotation DX", "Location", "best")