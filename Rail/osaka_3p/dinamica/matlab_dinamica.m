%% 
load("/Users/cristianvoltan/Desktop/unipd/tirocinio/CAD/Rail/osaka_3p/dinamica/dati_porta.mat")

idx_sx = porta == 0;
idx_dx = porta == 1;

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

% ══════════════════════════════
%  PORTA SINISTRA
% ══════════════════════════════
figure
sgtitle('Porta SINISTRA')

t_sx = tempo(idx_sx);

subplot(4,1,1)
plot(t_sx, apertura(idx_sx), 'b', 'LineWidth', 1.5)
ylabel('Door opening [deg]')
grid on
xlim([min(t_sx), max(t_sx)])

subplot(4,1,2)
plot(t_sx, velocita(idx_sx), 'b', 'LineWidth', 1.5)
ylabel('Angular velocity [deg/s]')
grid on
xlim([min(t_sx), max(t_sx)])

subplot(4,1,3)

tau_sx = tau(idx_sx,:);

hold on
plot(t_sx, tau_sx(:,1), 'Color', c1, 'LineWidth', 1.2)
plot(t_sx, tau_sx(:,2), 'Color', c2, 'LineWidth', 1.2)
plot(t_sx, tau_sx(:,3), 'Color', c3, 'LineWidth', 1.2)
plot(t_sx, tau_sx(:,4), 'Color', c4, 'LineWidth', 1.2)
plot(t_sx, tau_sx(:,5), 'Color', c5, 'LineWidth', 1.2)
plot(t_sx, tau_sx(:,6), 'Color', c6, 'LineWidth', 1.2)

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

ylabel('Joint torque [Nm]')
grid on
xlim([min(t_sx), max(t_sx)])
legend('J1','J2','J3','J4','J5','J6')


% ══════════════════════════════
%  PORTA DESTRA
% ══════════════════════════════
figure
sgtitle('Porta DESTRA')

t_dx = tempo(idx_dx);

subplot(4,1,1)
plot(t_dx, apertura(idx_dx), 'r', 'LineWidth', 1.5)
ylabel('Door opening [deg]')
grid on
xlim([min(t_dx), max(t_dx)])


subplot(4,1,2)
plot(t_dx, velocita(idx_dx), 'r', 'LineWidth', 1.5)
ylabel('Angular velocity [deg/s]')
grid on
xlim([min(t_dx), max(t_dx)])


subplot(4,1,3)

tau_dx = tau(idx_dx,:);

hold on
plot(t_dx, tau_dx(:,1), 'Color', c1, 'LineWidth', 1.2)
plot(t_dx, tau_dx(:,2), 'Color', c2, 'LineWidth', 1.2)
plot(t_dx, tau_dx(:,3), 'Color', c3, 'LineWidth', 1.2)
plot(t_dx, tau_dx(:,4), 'Color', c4, 'LineWidth', 1.2)
plot(t_dx, tau_dx(:,5), 'Color', c5, 'LineWidth', 1.2)
plot(t_dx, tau_dx(:,6), 'Color', c6, 'LineWidth', 1.2)

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

ylabel('Joint torque [Nm]')
grid on
xlim([min(t_dx), max(t_dx)])
legend('J1','J2','J3','J4','J5','J6')