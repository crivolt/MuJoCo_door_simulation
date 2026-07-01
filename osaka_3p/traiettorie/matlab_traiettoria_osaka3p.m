
% Load data trajectory
data_traj = readmatrix('/Users/cristianvoltan/Desktop/unipd/tirocinio/CAD/osaka_3p/traiettorie/traiettoria_destra.csv');
X = data_traj(:, 1);
Y = data_traj(:, 2);
Z = data_traj(:, 3);

% load data circle
dati_cerchio = readtable('/Users/cristianvoltan/Desktop/unipd/tirocinio/CAD/osaka_3p/traiettorie/parametri_cerchio_destra.csv');
Xc = dati_cerchio.Centro_X;
Yc = dati_cerchio.Centro_Y;
R  = dati_cerchio.Raggio;

% generate circumference
theta = linspace(0, 2*pi, 100);
X_cerchio = Xc + R * cos(theta);
Y_cerchio = Yc + R * sin(theta);
Z_cerchio = ones(size(X_cerchio)) * mean(Z); 

% 3d plot
figure;
hold on;
grid on;
plot3(X_cerchio, Y_cerchio, Z_cerchio, '--k', 'LineWidth', 1.5)
plot3(X, Y, Z, 'b', 'LineWidth', 2);


% start and end points
plot3(X(1), Y(1), Z(1), 'go', 'MarkerSize', 10, 'MarkerFaceColor', 'g'); % Inizio (Verde)
plot3(X(end), Y(end), Z(end), 'ro', 'MarkerSize', 10, 'MarkerFaceColor', 'r'); % Fine (Rosso)

xlabel('X (m)');
ylabel('Y (m)');
zlabel('Z (m)');
title('Trajectory');
legend('Ideal Circumference', 'Trajectory', 'Start', 'End');
axis equal;

view(3)