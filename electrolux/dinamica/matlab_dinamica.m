
clear; clc; close all;

% ══════════════════════════════
%  CARICAMENTO DATI E SETUP
% ══════════════════════════════
load("/Users/cristianvoltan/Desktop/unipd/tirocinio/CAD/electrolux/dinamica/dati_porta.mat")

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
plot(tempo, apertura, 'LineWidth', 3.0)
grid on
xlabel("Time [s]", "FontSize", 40)
ylabel("Door angle [deg]", "FontSize", 40)
set(gca, "FontSize", 30)

% ══════════════════════════════
% FIGURE 2 - ANGULAR VELOCITY
% ══════════════════════════════
figure("Name", "Door angular velocity")
plot(tempo, velocita, 'LineWidth', 3.0)
grid on
xlabel("Time [s]", "FontSize", 40)
ylabel("Door angular velocity [deg/s]", "FontSize", 40)
set(gca, "FontSize", 30)

% ══════════════════════════════
% FIGURE 3 - JOINT TORQUES
% ══════════════════════════════
figure("Name", "Joint torques")
hold on
plot(tempo, tau(:,1), 'Color', c1, 'LineWidth', 3.0)
plot(tempo, tau(:,2), 'Color', c2, 'LineWidth', 3.0)
plot(tempo, tau(:,3), 'Color', c3, 'LineWidth', 3.0)
plot(tempo, tau(:,4), 'Color', c4, 'LineWidth', 3.0)
plot(tempo, tau(:,5), 'Color', c5, 'LineWidth', 3.0)
plot(tempo, tau(:,6), 'Color', c6, 'LineWidth', 3.0)

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

xlabel("Time [s]", "FontSize", 40)
ylabel("Joint torque [Nm]", "FontSize", 40)
legend("J1","J2","J3","J4","J5","J6", "Location", "best", "FontSize", 30)
grid on
xlim([min(tempo), max(tempo)])
ylim([-335 335])
set(gca, "FontSize", 30)

% load("/Users/cristianvoltan/Desktop/unipd/tirocinio/CAD/electrolux/dinamica/dati_porta_franka.mat")
% 
% % === COLORI CURVE ===
% c1 = [0.6 0.9 0.5];    % verde chiarissimo
% c2 = [0.4 0.75 0.35];  % verde chiaro
% c3 = [0.2 0.6 0.2];    % verde medio
% c4 = [0.0 0.4 0.0];    % verde scuro
% 
% c5 = [1.0 0.7 0.4];    % arancio chiaro
% c6 = [1.0 0.4 0.2];    % arancio-rosso
% c7 = [0.75 0.0 0.0];   % rosso scuro
% 
% darkGreen = [0 0.35 0];
% redLine   = [0.85 0 0];
% 
% LW_CURVE = 2.5;   % spessore curve
% LW_LIM   = 2.0;   % spessore linee limite
% LW_AXES  = 1.5;   % spessore riquadro/assi
% FS       = 20;    % font size numeri assi ed etichette
% FS_LEG   = 16;    % font size legenda
% 
% fig = figure('Units','centimeters','Position',[0 0 24 18]);
% 
% % ===== Joint torques J1-J4 =====
% subplot(2,1,1)
% hold on
% plot(tempo, tau(:,1), 'Color', c1, 'LineWidth', LW_CURVE)
% plot(tempo, tau(:,2), 'Color', c2, 'LineWidth', LW_CURVE)
% plot(tempo, tau(:,3), 'Color', c3, 'LineWidth', LW_CURVE)
% plot(tempo, tau(:,4), 'Color', c4, 'LineWidth', LW_CURVE)
% yline( 87, '--', 'Color', darkGreen, 'LineWidth', LW_LIM)
% yline(-87, '--', 'Color', darkGreen, 'LineWidth', LW_LIM)
% 
% ylabel('Torque [Nm]', 'FontSize', FS)
% xlabel('Time [s]', 'FontSize', FS)
% grid on
% box on
% set(gca, 'FontSize', FS, 'LineWidth', LW_AXES)
% legend('J1','J2','J3','J4', 'FontSize', FS_LEG, 'Location', 'best')
% 
% % ===== Joint torques J5-J7 =====
% subplot(2,1,2)
% hold on
% plot(tempo, tau(:,5), 'Color', c5, 'LineWidth', LW_CURVE)
% plot(tempo, tau(:,6), 'Color', c6, 'LineWidth', LW_CURVE)
% plot(tempo, tau(:,7), 'Color', c7, 'LineWidth', LW_CURVE)
% yline( 12, '--', 'Color', redLine, 'LineWidth', LW_LIM)
% yline(-12, '--', 'Color', redLine, 'LineWidth', LW_LIM)
% 
% ylabel('Torque [Nm]', 'FontSize', FS)
% xlabel('Time [s]', 'FontSize', FS)
% grid on
% box on
% set(gca, 'FontSize', FS, 'LineWidth', LW_AXES)
% legend('J5','J6','J7', 'FontSize', FS_LEG, 'Location', 'best')
% 
% exportgraphics(fig, 'Franka_torques.png', 'Resolution', 300)
