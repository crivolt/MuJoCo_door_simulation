% Load data
dati = readmatrix('/Users/cristianvoltan/Desktop/unipd/tirocinio/CAD/osaka/traiettorie/traiettoria_sinistra.csv');


X = dati(:, 1);
Y = dati(:, 2);
Z = dati(:, 3);

% 3d plot
figure ('Units', 'centimeters', 'Position', [2 2 18 14], 'Color', 'w');
hold on;
grid on;
box on;
plot3(X, Y, Z, 'b', 'LineWidth', 3);

% start and end points
plot3(X(1), Y(1), Z(1), 'go', 'MarkerSize', 14, 'MarkerFaceColor', 'g', 'LineWidth', 1.2); % Inizio (Verde)
plot3(X(end), Y(end), Z(end), 'ro', 'MarkerSize', 14, 'MarkerFaceColor', 'r', 'LineWidth', 1.2); % Fine (Rosso)

ax = gca;
ax.FontSize = 28;
ax.LineWidth = 2.0;

% Labels and title
xlabel('X (m)', 'FontSize', 32, 'FontWeight', 'bold');
ylabel('Y (m)', 'FontSize', 32, 'FontWeight', 'bold');
zlabel('Z (m)', 'FontSize', 32, 'FontWeight', 'bold');
title('Trajectory', 'FontSize', 36, 'FontWeight', 'bold');
legend('Trajectory', 'Start', 'End', 'FontSize', 28, 'Location', 'northeast', 'Box', 'on');
axis equal;

view(3)
