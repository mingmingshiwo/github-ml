pwd;
d = load('ex1data1.txt');
X=d(:,1);
y=d(:,2);
y2=sin(X);
%plot(X,y,'rx', X, y2 ,'bx');

% m = length(X);
% X = [ones(m,1),X];
% theta = zeros(2,1);
% %cost = sum((X*theta-y).^2)/(2*m)
% temp = (X*theta - y);
% cost = (temp' * temp)/(2*m)

m = length(X);
X = [ones(m,1),X];
theta = zeros(2,1);
alpha = 0.01;
iteration = 1500;

%gradient descnet
[rst1,history1] = gradientDescent(X,y,theta,alpha,iteration);
% [rst2,history2] = gradientDescent(X,y,theta,alpha,100000);
% [rst3,history3] = gradientDescent(X,y,theta,0.02,100000);
figure(1);
plot(history1,'r');
% figure(2);
% plot(history2,'r');
% figure(3);
% plot(history3,'r');


theta_0 = linspace(-10,10,100);
theta_1 = linspace(-1,4,100);
cost_metrix = zeros(length(theta_0),length(theta_1));

for i = 1:length(theta_0)
    for j = 1:length(theta_1);
        t = [theta_0(i);theta_1(j)];
        cost_metrix(i,j) = computeCost(X, y,t);
    end
end
cost_metrix = cost_metrix';
surf(theta_0,theta_1,cost_metrix);

figure;
contour(theta_0,theta_1,cost_metrix,logspace(-2,3,20));
xlabel('\theta_0'); ylabel('\theta_1');
hold on;
plot(rst1(1), rst1(2), 'rx', 'MarkerSize', 10, 'LineWidth', 2);