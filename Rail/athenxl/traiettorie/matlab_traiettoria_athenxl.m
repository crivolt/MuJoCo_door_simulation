% Load data
dati = readmatrix('/Users/cristianvoltan/Desktop/unipd/tirocinio/CAD/athenxl/traiettorie/traiettoria_sinistra.csv');

X = dati(:, 1);
Y = dati(:, 2);
Z = dati(:, 3);

% 3d plot
figure;
plot3(X, Y, Z, 'b', 'LineWidth', 2);
hold on;
grid on;

% start and end points
plot3(X(1), Y(1), Z(1), 'go', 'MarkerSize', 10, 'MarkerFaceColor', 'g'); % Inizio (Verde)
plot3(X(end), Y(end), Z(end), 'ro', 'MarkerSize', 10, 'MarkerFaceColor', 'r'); % Fine (Rosso)

xlabel('X (m)');
ylabel('Y (m)');
zlabel('Z (m)');
title('Trajectory');
legend('Trajectory', 'Start', 'End');


