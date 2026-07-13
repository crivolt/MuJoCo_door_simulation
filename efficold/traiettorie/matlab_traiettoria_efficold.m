% Load data
dati_traj = readmatrix('/Users/cristianvoltan/Desktop/unipd/tirocinio/CAD/efficold/traiettorie/traiettoria_porta.csv');
X = dati_traj(:, 1);
Y = dati_traj(:, 2);
Z = dati_traj(:, 3);

% load data circle
dati_cerchio = readtable('/Users/cristianvoltan/Desktop/unipd/tirocinio/CAD/efficold/traiettorie/parametri_cerchio.csv');
Xc = dati_cerchio.Centro_X;
Yc = dati_cerchio.Centro_Y;
Zc = dati_cerchio.Centro_Z;
R  = dati_cerchio.Raggio;

% generate circonference
theta = linspace(0, 2*pi, 100);
X_cerchio = ones(size(theta)) * Xc; 
Y_cerchio = Yc + R * cos(theta);   
Z_cerchio = Zc + R * sin(theta);

% Create figure
figure('Units', 'centimeters', 'Position', [2 2 14 11], 'Color', 'w');

hold on;
grid on;
box on;

% Plot trajectory and ideal circumference
plot3(X, Y, Z, 'b', 'LineWidth', 3);
plot3(X_cerchio, Y_cerchio, Z_cerchio, '--k', 'LineWidth', 2.5);

% Start and end points
plot3(X(1), Y(1), Z(1), 'go', 'MarkerSize', 14, 'MarkerFaceColor', 'g', 'LineWidth', 1.2);
plot3(X(end), Y(end), Z(end), 'ro', 'MarkerSize', 14, 'MarkerFaceColor', 'r', 'LineWidth', 1.2);

% Axis appearance
ax = gca;
ax.FontSize = 20;
ax.LineWidth = 1.5;

% Labels and title
xlabel('X (m)', 'FontSize', 25, 'FontWeight', 'bold');
ylabel('Y (m)', 'FontSize', 25, 'FontWeight', 'bold');
zlabel('Z (m)', 'FontSize', 25, 'FontWeight', 'bold');
title('Trajectory', 'FontSize', 30, 'FontWeight', 'bold');

% Legend
legend('Trajectory', 'Ideal Circumference', 'Start', 'End', 'FontSize', 20, 'Location', 'northeast', 'Box', 'on');

axis equal;



