%rng(1);

d = 1;              % Spatial dimension

L = 2;              % Number of hidden layers
H = 200*[1,1,1];   % Layer widths
N = 1;            % Sample on N or N*N grid

% Stable parameters
par.alpha = 2;
par.beta = 0;
par.gamma = 1;
par.delta = 0;

if d==1
    XX = linspace(-1,1,N);
else
    [X,Y] = meshgrid(linspace(-1,1,N));
    XX = [X(:),Y(:)]';
end

% Activation function and derivative
sig = @(z) erf(z); 
dsig = @(z) 2*sqrt(2)*normpdf(z*sqrt(2));

% Cost function and derivative
cost = @(a) 0.5*norm(a-1).^2;
dcost = @(a) (a-1);

% Generate the weights
W = cell(L+1,1);
B = cell(L,1);

W{1} = stblrnd(par.alpha,par.beta,par.gamma,par.delta,H(1),d);
B{1} = stblrnd(par.alpha,par.beta,par.gamma,par.delta,H(1),1);

NW = d*H(1) + H(1);
for l=2:L
    W{l} = stblrnd(par.alpha,par.beta,par.gamma,par.delta,H(l),H(l-1));
    B{l} = stblrnd(par.alpha,par.beta,par.gamma,par.delta,H(l),1);
    NW = NW + H(l)*H(l-1) + H(l);
end
W{L+1} = stblrnd(par.alpha,par.beta,par.gamma,par.delta,1,H(L));
NW = NW + H(L);

dC = cell(N,1);
for n = 1:N
    % Build network
    z = cell(L+1,1);
    a = cell(L+1,1);

    z{1} = W{1}*XX(n) + B{1};
    a{1} = sig(z{1});
    for l=2:L
        z{l} = W{l}*a{l-1} + B{l};
        a{l} = sig(z{l});
    end
    z{L+1} = W{L+1}*a{L};
    a{L+1} = z{L+1};

    % Calculate cost derivative (backprop);
    delta = cell(L+1,1);
    delta{L+1} = dcost(a{L+1});

    for l=L:-1:1
       delta{l} = (W{l+1}'*delta{l+1}).*dsig(z{l}); 
    end

    temp = delta{1}.*XX(n)';
    dC{n} = temp(:);
    dC{n} = [dC{n};delta{1}];
    for l=2:L
       temp = delta{l}*a{l-1}';
       dC{n} = [dC{n};temp(:)];
       dC{n} = [dC{n};delta{l}];
    end
    temp = delta{L+1}*a{L}';
    dC{n} = [dC{n};temp(:)];

end

% Collect Jacobian
dCF = zeros(NW,N);
for n=1:N
    dCF(:,n) = dC{n};
end

%surf(dCF,'EdgeColor','None')


%
% fd test
ep = 1e-10;

c0 = cost(a{L+1});

ind = 1;
WP = W;
WP{1}(ind) = WP{1}(ind) + ep;

z = cell(L+1,1);
a = cell(L+1,1);

z{1} = WP{1}*XX + B{1};
a{1} = sig(z{1});
for l=2:L
    z{l} = WP{l}*a{l-1} + B{l};
    a{l} = sig(z{l});
end
z{L+1} = WP{L+1}*a{L};
a{L+1} = z{L+1};

cP = cost(a{L+1});

disp(dC{1}(ind));
disp((cP-c0)/ep);
%

