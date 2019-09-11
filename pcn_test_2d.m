S = 1000000;
beta = 0.01;

local = 100; % how many steps to calculate local acceptance rate?
thinning = 1000; % how often to update figure?
burnin = 100000;

H0 = 100;
N = 2^3;            % Sample on N or N*N grid
J = N^2;
gm = 0.04;

NW = 5*H0+H0^2;

[X,Y] = meshgrid(linspace(-1,1,N));
XX = [X(:),Y(:)]';

UT = (0.5*(sign(X+0.5)-sign(X-0.5))-0.5)';
UT = UT(:);

obsInd = randsample(N^2,J);
basis = speye(N^2);
A = basis(obsInd,:);

y = A*UT + normrnd(0,1,J,1)*gm;

Phi = @(xi) norm(A*build_net(xi,XX)-y)^2/(2*gm^2);

subplot(131);
surf(X,Y,reshape(UT,N,N),'EdgeColor','None');view(2);axis square;colorbar;
hold on
scatter3(XX(1,obsInd),XX(2,obsInd),10*ones(J,1),'x');
hold off

acc = zeros(S,1);

UMean = zeros(N^2,1);
UVar = zeros(N^2,1);

xi = normrnd(0,1,NW,1);
build_net(xi,XX);

for s=1:S
    xiProp = sqrt(1-beta^2)*xi +beta*normrnd(0,1,NW,1);
    aProb = min(1,exp(Phi(xi)-Phi(xiProp)));
    if unifrnd(0,1) < aProb
        xi = xiProp;
        acc(s) = 1;
    end
    
    if mod(s,local) == 0
        l_acc = mean(acc(s-local+1:s));
        if l_acc < 0.3
            beta = beta/2;
        else
            beta = min(1,beta*2);
        end
    end
        
    if mod(s,thinning) == 0           
        U = build_net(xi,XX);
        
        if s > burnin
            q = (s-burnin)/thinning;
            UMean = ((q-1)*UMean + U)/q;
            UVar = ((q-1)*UVar + U.^2)/q;

            subplot(132);
            surf(X,Y,reshape(UMean,N,N),'EdgeColor','None');view(2);axis square;colorbar;
        end
        
        subplot(133);
        surf(X,Y,reshape(U,N,N),'EdgeColor','None');view(2);axis square;colorbar;
        pause(0.01);
        fprintf('Accepted: %f\n',l_acc);
    end
end




function f = build_net(xi,XX)
    d = 2;              % Spatial dimension
    T = @(z) tan(pi*normcdf(z)-pi/2);
    sig = @(z) tanh(z);  % Activation function

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

    

