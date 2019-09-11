S = 1000000;
beta = 0.01;

local = 100; % how many steps to calculate local acceptance rate?
thinning = 1000; % how often to update figure?
burnin = 100000;

H0 = 100;
N = 2^8;            % Sample on N or N*N grid
J = 50;
gm = 0.04;

NW = 4*H0+H0^2;

XX = linspace(-1,1,N);

UT = (0.5*(sign(XX+0.5)-sign(XX-0.5))-0.5)'+cos(4*pi*XX').*(XX'>0.5);

obsInd = randsample(N,J);
basis = speye(N);

B = zeros(N);
for j=1:N
    B(:,j) = idct(dct(full(basis(:,j))).*exp(-0.1*(1:N)'));
end

A = basis(obsInd,:);

y = A*B*UT + normrnd(0,1,J,1)*gm;

Phi = @(xi) norm(A*B*build_net(xi,XX)-y)^2/(2*gm^2);

subplot(131);
plot(XX,UT);axis([-1,1,-2,1]);
hold on
scatter(XX(obsInd),y,'x');
hold off

acc = zeros(S,1);

UMean = zeros(N,1);
UVar = zeros(N,1);

xi = normrnd(0,1,NW,1);
PhiXi = Phi(xi);
for s=1:S
    xiProp = sqrt(1-beta^2)*xi +beta*normrnd(0,1,NW,1);
    PhiXiProp = Phi(xiProp);
    aProb = min(1,exp(PhiXi-PhiXiProp));
    if unifrnd(0,1) < aProb
        xi = xiProp;
        PhiXi = PhiXiProp;
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
            plot(XX,[UMean,UMean+sqrt(UVar-UMean.^2),UMean-sqrt(UVar-UMean.^2)]);axis([-1,1,-2,1]);
        end
        
        subplot(133);
        plot(XX,U);axis([-1,1,-2,1]);
        pause(0.01);
        fprintf('s = %i\tAccepted: %f\n',s,l_acc);
    end
end




function f = build_net(xi,XX)
    d = 1;              % Spatial dimension
    T = @(z) tan(pi*normcdf(z)-pi/2);
    %T = @(z) z;
    sig = @(z) erf(z);  % Activation function

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

    

