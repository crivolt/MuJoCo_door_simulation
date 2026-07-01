
load("/Users/cristianvoltan/Desktop/unipd/tirocinio/CAD/studio_54/dinamica/dati_porta.mat")

figure

% Door opening
subplot(3,1,1)
plot(tempo, apertura, 'LineWidth', 1.5)
ylabel('Door opening [deg]')
title('Door opening, velocity and joint torques')
grid on

% Door angular velocity
subplot(3,1,2)
plot(tempo, velocita, 'LineWidth', 1.5)
ylabel('Angular velocity [deg/s]')
grid on

% Joint torques
subplot(3,1,3)

% === COLORI CURVE ===
c1 = [0.4 0.8 0.4];    % verde chiaro
c2 = [0.0 0.6 0.0];    % verde scuro
c3 = [0.4 0.8 1.0];    % azzurro

c4 = [1.0 0.6 0.0];    % arancione
c5 = [1.0 0.3 0.0];    % arancio-rosso
c6 = [0.9 0.1 0.1];    % rosso scuro

plot(tempo, tau(:,1), 'Color', c1, 'LineWidth', 1.2)
hold on
plot(tempo, tau(:,2), 'Color', c2, 'LineWidth', 1.2)
plot(tempo, tau(:,3), 'Color', c3, 'LineWidth', 1.2)
plot(tempo, tau(:,4), 'Color', c4, 'LineWidth', 1.2)
plot(tempo, tau(:,5), 'Color', c5, 'LineWidth', 1.2)
plot(tempo, tau(:,6), 'Color', c6, 'LineWidth', 1.2)

limits = [330, 330, 150, 56, 56, 56];

darkGreen = [0 0.35 0];  % limiti J1-J2
blueLine  = [0 0 0.8];   % limite J3
redLine   = [0.85 0 0];  % limiti J4-J6 (rosso)

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
xlabel('Time [s]')
grid on
legend('J1','J2','J3','J4','J5','J6')
