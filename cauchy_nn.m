d = 2;              % Spatial dimension
S = 1;

L =4;              % Number of layers
H = [625,10,300,150];   % Layer widths
%H = [625,400,324];
N = 2^9;            % Sample on N or N*N grid

% Stable parameters
par.alpha = 2;
par.beta = 0;
par.gamma = 1;
par.delta = 0;

if d==1
    XX = linspace(-1,1,N);
else
    [X,Y] = meshgrid(linspace(0,1,N));
    XX = [X(:),Y(:)]';
end

sig = @(z) ReLU(z);  % Activation function

U = zeros(1,N);
U2 = zeros(1,N);
for s=1:S
    W = cell(L,1);
    A = cell(L,1);

    W{1} = sp(stblrnd(par.alpha,par.beta,par.gamma,par.delta,d,H(1)));
    A{1} = sp(stblrnd(par.alpha,par.beta,par.gamma,par.delta,H(1),1));

    NW = d*H(1) + H(1);
    for l=2:L
        W{l} = sp(stblrnd(par.alpha,par.beta,par.gamma,par.delta,H(l-1),H(l)));
        A{l} = sp(stblrnd(par.alpha,par.beta,par.gamma,par.delta,H(l),1));
        NW = NW + H(l)*H(l-1) + H(l);
    end

    V = sp(stblrnd(par.alpha,par.beta,par.gamma,par.delta,H(L),1));
    NW = NW + H(L);

    h = sig(A{1}+W{1}'*XX);
    for l=2:L
        h = sig(A{l} + W{l}'*h);
    end
    f = (stblrnd(par.alpha,par.beta,par.gamma,par.delta)*0 + V'*h)/H(L)^(1/par.alpha);
    %U = ((s-1)*U + abs(gradient(f)))/s;
    %U2 = ((s-1)*U2 +gradient(f).^2)/s;
end

if d==1
    plot(XX,f);
else
    surf(X,Y,reshape(f,N,N),'EdgeColor','None');view(2);axis square;
    %colorbar;
    colormap Jet;
end
pause(0.01);

function B = sp(A)
    B = A;%sparse(A.*(abs(A)>0.0));
end