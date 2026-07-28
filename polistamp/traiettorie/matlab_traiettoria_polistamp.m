% Load data
dati_traj = readmatrix('/Users/cristianvoltan/Desktop/unipd/tirocinio/CAD/polistamp/traiettorie/traiettoria_apertura_maniglia.csv');

X = dati_traj(:, 1);
Y = dati_traj(:, 2);
Z = dati_traj(:, 3);

% load data circle
dati_cerchio = readtable('/Users/cristianvoltan/Desktop/unipd/tirocinio/CAD/polistamp/traiettorie/parametri_cerchio_maniglia.csv');
Xc = dati_cerchio.Centro_X;
Yc = dati_cerchio.Centro_Y;
R  = dati_cerchio.Raggio;

% generate circonference
theta = linspace(0, 2*pi, 100);
X_cerchio = Xc + R * cos(theta);
Y_cerchio = Yc + R * sin(theta);
Z_cerchio = ones(size(X_cerchio)) * mean(Z);

% 3d plot
figure ('Units', 'centimeters', 'Position', [2 2 18 14], 'Color', 'w');
hold on;
grid on;
box on;
plot3(X, Y, Z, 'b', 'LineWidth', 3);
plot3(X_cerchio, Y_cerchio, Z_cerchio, '--k', 'LineWidth', 2.5)

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
legend('Trajectory', 'Ideal Circumference', 'Start', 'End', 'FontSize', 28, 'Location', 'northeast', 'Box', 'on');
axis equal;

view(3)


