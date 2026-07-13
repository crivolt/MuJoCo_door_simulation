% Load data
dati = readmatrix('/Users/cristianvoltan/Desktop/unipd/tirocinio/CAD/osaka/traiettorie/traiettoria_destra.csv');


X = dati(:, 1);
Y = dati(:, 2);
Z = dati(:, 3);

% 3d plot
figure ('Units', 'centimeters', 'Position', [2 2 14 11], 'Color', 'w');
hold on;
grid on;
box on;
plot3(X, Y, Z, 'b', 'LineWidth', 3);

% start and end points
plot3(X(1), Y(1), Z(1), 'go', 'MarkerSize', 14, 'MarkerFaceColor', 'g', 'LineWidth', 1.2); % Inizio (Verde)
plot3(X(end), Y(end), Z(end), 'ro', 'MarkerSize', 14, 'MarkerFaceColor', 'r', 'LineWidth', 1.2); % Fine (Rosso)

ax = gca;
ax.FontSize = 20;
ax.LineWidth = 1.5;

% Labels and title
xlabel('X (m)', 'FontSize', 25, 'FontWeight', 'bold');
ylabel('Y (m)', 'FontSize', 25, 'FontWeight', 'bold');
zlabel('Z (m)', 'FontSize', 25, 'FontWeight', 'bold');
title('Trajectory', 'FontSize', 30, 'FontWeight', 'bold');
legend('Trajectory', 'Start', 'End', 'FontSize', 20, 'Location', 'northeast', 'Box', 'on');
axis equal;

view(3)
