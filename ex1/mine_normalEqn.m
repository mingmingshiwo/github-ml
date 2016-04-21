d = load('ex1data1.txt');
X=d(:,1);
y=d(:,2);
m = length(X);
X = [ones(m,1),X];
theta = pinv(X'*X)*X'*y
theta_ = inv(X'*X)*X'*y