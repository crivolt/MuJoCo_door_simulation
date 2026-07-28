clear
clc
close all

% Load data
data = load("/Users/cristianvoltan/Desktop/unipd/tirocinio/CAD/electric_solution/osaka/dati_porta_lineare_athen_xl.mat");

t = data.tempo(:);

% Porta sinistra
door_pos_sx = data.posizione_porta_sx_m(:);
door_vel_sx = data.velocita_porta_sx_m_s(:);
act_pos_sx = data.posizione_attuatore_lineare_sx_m(:);
act_vel_sx = data.velocita_attuatore_lineare_sx_m_s(:);
align_rot_sx = data.rotazione_allineamento_sx_deg(:);

% Porta destra
door_pos_dx = data.posizione_porta_dx_m(:);
door_vel_dx = data.velocita_porta_dx_m_s(:);
act_pos_dx = data.posizione_attuatore_lineare_dx_m(:);
act_vel_dx = data.velocita_attuatore_lineare_dx_m_s(:);
align_rot_dx = data.rotazione_allineamento_dx_deg(:);


% Figure 1 - Door position
figure("Name", "Door position")
plot(t, door_pos_sx, "LineWidth", 1.5)
hold on
plot(t, door_pos_dx, "LineWidth", 1.5)
grid on
xlabel("Time [s]", "FontSize", 30)
ylabel("Door displacement [m]", "FontSize", 30)
title("Door Position Over Time", "FontSize", 36)
legend("Door SX", "Door DX", "Location", "best", "FontSize", 30)
set(gca, "FontSize", 25)

% Figure 2 - Door velocity
figure("Name", "Door velocity")
plot(t, door_vel_sx, "LineWidth", 1.5)
hold on
plot(t, door_vel_dx, "LineWidth", 1.5)
grid on
xlabel("Time [s]", "FontSize", 30)
ylabel("Door velocity [m/s]", "FontSize", 30)
title("Door Velocity Over Time", "FontSize", 36)
legend("Door SX", "Door DX", "Location", "best", "FontSize", 30)
set(gca, "FontSize", 25)

% Figure 3 - Linear actuator position
figure("Name", "Linear actuator position")
plot(t, act_pos_sx, "LineWidth", 1.5)
hold on
plot(t, act_pos_dx, "LineWidth", 1.5)
grid on
xlabel("Time [s]", "FontSize", 30)
ylabel("Actuator displacement [m]", "FontSize", 30)
title("Linear Actuator Position Over Time", "FontSize", 36)
legend("Actuator SX", "Actuator DX", "Location", "best", "FontSize", 30)
set(gca, "FontSize", 25)

% Figure 4 - Linear actuator velocity
figure("Name", "Linear actuator velocity")
plot(t, act_vel_sx, "LineWidth", 1.5)
hold on
plot(t, act_vel_dx, "LineWidth", 1.5)
grid on
xlabel("Time [s]", "FontSize", 30)
ylabel("Actuator velocity [m/s]", "FontSize", 30)
title("Linear Actuator Velocity Over Time", "FontSize", 36)
legend("Actuator SX", "Actuator DX", "Location", "best", "FontSize", 30)
set(gca, "FontSize", 25)

% Figure 5 - Alignment joint rotation
figure("Name", "Alignment joint rotation")
plot(t, align_rot_sx, "LineWidth", 1.5)
hold on
plot(t, align_rot_dx, "LineWidth", 1.5)
grid on
xlabel("Time [s]", "FontSize", 30)
ylabel("Rotation [deg]", "FontSize", 30)
title("Alignment Joint Rotation Over Time", "FontSize", 36)
legend("Alignment SX", "Alignment DX", "Location", "best", "FontSize", 30)
set(gca, "FontSize", 25)


% Figure 6 - Complete overview
figure("Name", "Complete overview")
tiledlayout(5, 1)

nexttile
plot(t, door_pos_sx, "LineWidth", 1.5)
hold on
plot(t, door_pos_dx, "LineWidth", 1.5)
grid on
ylabel("Displacement [m]")
title("Door Position")
legend("Door SX", "Door DX", "Location", "best")

nexttile
plot(t, door_vel_sx, "LineWidth", 1.5)
hold on
plot(t, door_vel_dx, "LineWidth", 1.5)
grid on
ylabel("Velocity [m/s]")
title("Door Velocity")
legend("Door SX", "Door DX", "Location", "best")

nexttile
plot(t, act_pos_sx, "LineWidth", 1.5)
hold on
plot(t, act_pos_dx, "LineWidth", 1.5)
grid on
ylabel("Displacement [m]")
title("Linear Actuator Position")
legend("Actuator SX", "Actuator DX", "Location", "best")

nexttile
plot(t, act_vel_sx, "LineWidth", 1.5)
hold on
plot(t, act_vel_dx, "LineWidth", 1.5)
grid on
ylabel("Velocity [m/s]")
title("Linear Actuator Velocity")
legend("Actuator SX", "Actuator DX", "Location", "best")

nexttile
plot(t, align_rot_sx, "LineWidth", 1.5)
hold on
plot(t, align_rot_dx, "LineWidth", 1.5)
grid on
xlabel("Time [s]")
ylabel("Rotation [deg]")
title("Alignment Joint Rotation")
legend("Rotation SX", "Rotation DX", "Location", "best")