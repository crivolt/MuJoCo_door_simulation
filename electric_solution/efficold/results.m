clear
clc
close all

% Load data
data = load("/Users/cristianvoltan/Desktop/unipd/tirocinio/CAD/electric_solution/efficold/dati_porta_meccanica_efficold.mat");
t = data.tempo(:);
door_pos = data.posizione_porta(:);
door_vel = data.velocita_porta(:);
motor_pos = data.posizione_motoriduttore(:);
comp_radial = data.correzione_allineamento_radiale_mm(:);
align_rot = data.rotazione_allineamento_deg(:);

% Figure 1 - Door position
figure("Name", "Door position")
plot(t, door_pos, "LineWidth", 1.5)
grid on
xlabel("Time [s]", "FontSize", 30)
ylabel("Door angle [deg]", "FontSize", 30)
title("Door Position Over Time", "FontSize", 36)
legend("Door", "Location", "best", "FontSize", 30)
set(gca, "FontSize", 25)

% Figure 2 - Door velocity
figure("Name", "Door velocity")
plot(t, door_vel, "LineWidth", 1.5)
grid on
xlabel("Time [s]", "FontSize", 30)
ylabel("Door velocity [deg/s]", "FontSize", 30)
title("Door Velocity Over Time", "FontSize", 36)
legend("Door", "Location", "best", "FontSize", 30)
set(gca, "FontSize", 25)

% Figure 3 - Gearmotor position
figure("Name", "Gearmotor position")
plot(t, motor_pos, "LineWidth", 1.5)
grid on
xlabel("Time [s]", "FontSize", 30)
ylabel("Motor angle [deg]", "FontSize", 30)
title("Gearmotor Position Over Time", "FontSize", 36)
legend("Motor", "Location", "best", "FontSize", 30)
set(gca, "FontSize", 25)

% Figure 4 - Alignment corrections
figure("Name", "Alignment corrections")
plot(t, comp_radial, "LineWidth", 1.5)
hold on
plot(t, align_rot, "LineWidth", 1.5)
grid on
xlabel("Time [s]", "FontSize", 30)
ylabel("Correction", "FontSize", 30)
title("Alignment Corrections Over Time", "FontSize", 36)
legend("Radial correction [mm]", "Rotation [deg]", "Location", "best", "FontSize", 30)
set(gca, "FontSize", 25)

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