S = 1000000;
beta = 0.01;

local = 100; % how many steps to calculate local acceptance rate?
thinning = 1000; % how often to update figure?
burnin = 20000;

H0 = 100;
N = 2^10;            % Sample on N or N*N grid
J = N^2;
gm = 0.04;

NW = 5*H0+H0^2;

[X,Y] = meshgrid(linspace(-1,1,N));
XX = [X(:),Y(:)]';

UT = (0.5*(sign(X+0.5)-sign(X-0.5))-0.5)';
UT = UT(:);


U = build_net(xi,XX);

surf(X,Y,reshape(U,N,N),'EdgeColor','None');view(2);axis square;colorbar;
return



function f = build_net(xi,XX)
    d = 2;              % Spatial dimension
    T = @(z) tan(pi*normcdf(z)-pi/2);
    sig = @(z) reluLayer(z);  % Activation function

    H0 = 100;
    
    W = cell(2,1);
    A = cell(2,1);

    W{1} = T(reshape(xi(1:d*H0),d,H0));
    A{1} = T(reshape(xi(d*H0+1:d*H0+H0),H0,1));

    W{2} = T(reshape(xi(d*H0+H0+1:d*H0+H0+H0^2),H0,H0));
    A{2} = T(reshape(xi(d*H0+H0+H0^2+1:d*H0+H0+H0^2+H0),H0,1));

    V = T(reshape(xi(d*H0+H0+H0^2+H0+1:end),H0,1));

    h = sig(A{1}+W{1}'*XX);
    h = sig(A{2} + W{2}'*h)/H0;
    f = (V'*h)';
end

    

