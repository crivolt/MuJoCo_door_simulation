clear; clc; close all;

% ══════════════════════════════
%  CARICAMENTO DATI E SETUP
% ══════════════════════════════
load("/Users/cristianvoltan/Desktop/unipd/tirocinio/CAD/Rail/osaka_3p/dinamica/dati_porta.mat")

idx_sx = porta == 0;
idx_dx = porta == 1;

% Vettori tempo ritagliati (servono per i subplot delle coppie)
t_sx = tempo(idx_sx);
t_dx = tempo(idx_dx);

limits = [330, 330, 150, 56, 56, 56];

% === COLORI CURVE ===
c1 = [0.4 0.8 0.4];    % verde chiaro
c2 = [0.0 0.6 0.0];    % verde scuro
c3 = [0.4 0.8 1.0];    % azzurro
c4 = [1.0 0.6 0.0];    % arancione
c5 = [1.0 0.3 0.0];    % arancio-rosso
c6 = [0.9 0.1 0.1];    % rosso scuro
darkGreen = [0 0.35 0];
blueLine  = [0 0 0.8];
redLine   = [0.85 0 0];

% === CREAZIONE ARRAY COMPLETI (PARTENZA DA ZERO) ===
% Creiamo array pieni di zeri lunghi quanto tutto il vettore 'tempo'
apertura_sx_full = zeros(size(tempo));
apertura_dx_full = zeros(size(tempo));
velocita_sx_full = zeros(size(tempo));
velocita_dx_full = zeros(size(tempo));
slitta_sx_full = zeros(size(tempo));
slitta_dx_full = zeros(size(tempo));

% Inseriamo i dati reali solo negli indici corretti
apertura_sx_full(idx_sx) = apertura(idx_sx);
apertura_dx_full(idx_dx) = apertura(idx_dx);
velocita_sx_full(idx_sx) = velocita(idx_sx);
velocita_dx_full(idx_dx) = velocita(idx_dx);
slitta_sx_full(idx_sx) = slitta(idx_sx);
slitta_dx_full(idx_dx) = slitta(idx_dx);

% ══════════════════════════════
% FIGURE 1 - DOOR OPENING (SX + DX)
% ══════════════════════════════
figure("Name", "Door opening angle")
plot(tempo, apertura_sx_full, 'LineWidth', 3.0)
hold on
plot(tempo, apertura_dx_full, 'LineWidth', 3.0)
grid on
xlabel("Time [s]", "FontSize", 30)
ylabel("Door angle [deg]", "FontSize", 30)
title("Door Opening Angle Over Time", "FontSize", 36)
legend("Door SX", "Door DX", "Location", "best", "FontSize", 30)
set(gca, "FontSize", 25)

% ══════════════════════════════
% FIGURE 2 - ANGULAR VELOCITY (SX + DX)
% ══════════════════════════════
figure("Name", "Door angular velocity")
plot(tempo, velocita_sx_full, 'LineWidth', 3.0)
hold on
plot(tempo, velocita_dx_full, 'LineWidth', 3.0)
grid on
xlabel("Time [s]", "FontSize", 30)
ylabel("Angular angular velocity [deg/s]", "FontSize", 30)
title("Door Angular Velocity Over Time", "FontSize", 36)
legend("Door SX", "Door DX", "Location", "best", "FontSize", 30)
set(gca, "FontSize", 25)

% ══════════════════════════════
% FIGURE 3 - JOINT TORQUES (SX e DX come subplot)
% ══════════════════════════════
figure("Name", "Joint torques")
tau_sx = tau(idx_sx,:);
tau_dx = tau(idx_dx,:);

subplot(2,1,1)
hold on
% NOTA: qui si usa t_sx per evitare errori di dimensione
plot(t_sx, tau_sx(:,1), 'Color', c1, 'LineWidth', 3.0)
plot(t_sx, tau_sx(:,2), 'Color', c2, 'LineWidth', 3.0)
plot(t_sx, tau_sx(:,3), 'Color', c3, 'LineWidth', 3.0)
plot(t_sx, tau_sx(:,4), 'Color', c4, 'LineWidth', 3.0)
plot(t_sx, tau_sx(:,5), 'Color', c5, 'LineWidth', 3.0)
plot(t_sx, tau_sx(:,6), 'Color', c6, 'LineWidth', 3.0)
for i = 1:6
    if i <= 2
        limColor = darkGreen;
    elseif i == 3
        limColor = blueLine;
    else
        limColor = redLine;
    end
    yline( limits(i), '--', 'Color', limColor, 'LineWidth', 2.0)
    yline(-limits(i), '--', 'Color', limColor, 'LineWidth', 2.0)
end
xlabel("Time [s]", "FontSize", 30)
ylabel("Joint torque [Nm]", "FontSize", 30)
title("Joint Torques - Door SX", "FontSize", 36)
legend("J1","J2","J3","J4","J5","J6", "Location", "best", "FontSize", 30)
grid on
xlim([min(t_sx), max(t_sx)])
ylim([-335 335])
set(gca, "FontSize", 25)

subplot(2,1,2)
hold on
% NOTA: qui si usa t_dx per evitare errori di dimensione
plot(t_dx, tau_dx(:,1), 'Color', c1, 'LineWidth', 3.0)
plot(t_dx, tau_dx(:,2), 'Color', c2, 'LineWidth', 3.0)
plot(t_dx, tau_dx(:,3), 'Color', c3, 'LineWidth', 3.0)
plot(t_dx, tau_dx(:,4), 'Color', c4, 'LineWidth', 3.0)
plot(t_dx, tau_dx(:,5), 'Color', c5, 'LineWidth', 3.0)
plot(t_dx, tau_dx(:,6), 'Color', c6, 'LineWidth', 3.0)
for i = 1:6
    if i <= 2
        limColor = darkGreen;
    elseif i == 3
        limColor = blueLine;
    else
        limColor = redLine;
    end
    yline( limits(i), '--', 'Color', limColor, 'LineWidth', 2.0)
    yline(-limits(i), '--', 'Color', limColor, 'LineWidth', 2.0)
end
xlabel("Time [s]", "FontSize", 30)
ylabel("Joint torque [Nm]", "FontSize", 30)
title("Joint Torques - Door DX", "FontSize", 36)
legend("J1","J2","J3","J4","J5","J6", "Location", "best", "FontSize", 30)
grid on
xlim([min(t_dx), max(t_dx)])
ylim([-335 335])
set(gca, "FontSize", 25)

% ══════════════════════════════
% FIGURE 4 - SLITTA (SX + DX)
% ══════════════════════════════
figure("Name", "Slitta position")
plot(tempo, slitta, 'LineWidth', 3.0)
grid on
xlabel("Time [s]", "FontSize", 30)
ylabel("Slitta position [m]", "FontSize", 30)
title("Slitta Position Over Time", "FontSize", 36)
set(gca, "FontSize", 25)