clear; clc; close all;

% ══════════════════════════════
%  CARICAMENTO DATI E SETUP
% ══════════════════════════════
load("/Users/cristianvoltan/Desktop/unipd/tirocinio/CAD/efficold/dinamica/dati_porta.mat")

limits = [330, 330, 150, 56, 56, 56];

% === COLORI CURVE ===
c1 = [0.4 0.8 0.4];    % verde chiaro
c2 = [0.0 0.6 0.0];    % verde scuro
c3 = [0.4 0.8 1.0];    % azzurro
c4 = [1.0 0.6 0.0];    % arancione
c5 = [1.0 0.3 0.0];    % arancio-rosso
c6 = [0.9 0.1 0.1];    % rosso scuro
darkGreen = [0 0.35 0];  % limiti J1-J2
blueLine  = [0 0 0.8];   % limite J3
redLine   = [0.85 0 0];  % limiti J4-J6 (rosso)

% ══════════════════════════════
% FIGURE 1 - DOOR OPENING ANGLE
% ══════════════════════════════
figure("Name", "Door opening angle")
plot(tempo, apertura, 'LineWidth', 1.5)
grid on
xlabel("Time [s]", "FontSize", 30)
ylabel("Door opening [deg]", "FontSize", 30)
title("Door Opening Angle Over Time", "FontSize", 36)
set(gca, "FontSize", 25)

% ══════════════════════════════
% FIGURE 2 - ANGULAR VELOCITY
% ══════════════════════════════
figure("Name", "Door angular velocity")
plot(tempo, velocita, 'LineWidth', 1.5)
grid on
xlabel("Time [s]", "FontSize", 30)
ylabel("Angular velocity [deg/s]", "FontSize", 30)
title("Door Angular Velocity Over Time", "FontSize", 36)
set(gca, "FontSize", 25)

% ══════════════════════════════
% FIGURE 3 - JOINT TORQUES
% ══════════════════════════════
figure("Name", "Joint torques")
hold on
plot(tempo, tau(:,1), 'Color', c1, 'LineWidth', 1.2)
plot(tempo, tau(:,2), 'Color', c2, 'LineWidth', 1.2)
plot(tempo, tau(:,3), 'Color', c3, 'LineWidth', 1.2)
plot(tempo, tau(:,4), 'Color', c4, 'LineWidth', 1.2)
plot(tempo, tau(:,5), 'Color', c5, 'LineWidth', 1.2)
plot(tempo, tau(:,6), 'Color', c6, 'LineWidth', 1.2)

for i = 1:6
    if i <= 2
        limColor = darkGreen;
    elseif i == 3
        limColor = blueLine;
    else
        limColor = redLine;
    end
    yline( limits(i), '--', 'Color', limColor, 'LineWidth', 1)
    yline(-limits(i), '--', 'Color', limColor, 'LineWidth', 1)
end

xlabel("Time [s]", "FontSize", 30)
ylabel("Joint torque [Nm]", "FontSize", 30)
title("Joint Torques", "FontSize", 36)
legend("J1","J2","J3","J4","J5","J6", "Location", "best", "FontSize", 30)
grid on
xlim([min(tempo), max(tempo)])
set(gca, "FontSize", 25)