%% Define the parameters
load('data_run.mat');

model = 'id';

% Sample resolution
%N = 2^3;							% Sampling performed on NxN square grid
NT = 2^8;						% Data generated on NTxNT square grid

% Prior parameters
prior.q = 1;
prior.s = 1;
prior.kappa = 0.1;

% Data parameters
data.J = 4^2;					% Dimension of observation space
data.gamma = 0.1;				% Noise standard deviation

% MCMC parameters
%mcmc.beta = 2^-4;				% pCN jump for U
mcmc.samples = 10000;			% Number of samples to generate

% Output parameters
output.figures = 1;				% Display figures?
output.save_figures = 0;		% Save figures?
output.thinning = 100;			% How often to update output?

betaSet = 0.01*2.^(-(0));
%betaSet = 0.01*ones(1,14);
NSet = 2.^(3:8);

acceptMean = zeros(12,6);



%% Create the data (or alternatively import from a file)

{
[K1T,K2T] = meshgrid(0:NT-1,0:NT-1);
rhoT = sqrt(prior.kappa)*(K1T.^2+K2T.^2).^(-prior.s);
rhoT(1) = 0;

buildT = @(U) NT*idct2(reshape(U,NT,NT).*rhoT);

T = @(xi) 2^(1/prior.q).*sign(xi).*abs(gammaincinv(min(2*normcdf(abs(xi))-1,1-1e-14),1/prior.q)).^(1/prior.q);

bnorm = @(U) sum(abs(U).^prior.q.*(rho.^prior.q));

xiT = normrnd(0,1,NT^2,1);
UT = buildT(T(xiT));

lower = 1.2*min(min(UT)); upper = 1.2*max(max(UT));

% Create the data
eta = normrnd(0,1,data.J,1);
data.clean = ell(UT,data.J,model);
data.noisy = data.clean + data.gamma*eta;
}


T = @(xi) -2*sign(xi).*log(max(abs(2-2*normcdf(abs(xi))),1e-14));
DT = @(xi) 2/sqrt(2*pi)*exp(-xi.^2/2)./(max(abs(1-normcdf(abs(xi))),1e-14));


TMax = 0.1;

for kk=1:length(NSet)
    
    N = NSet(kk);
    
    step = round(N/(sqrt(data.J)+1));
    obs_base = step:step:sqrt(data.J)*step;
    obsPts = obs_base + (step-1)*N;

    for p=2:sqrt(data.J)
        obsPts = [obsPts,obs_base + (step*p-1)*N];
    end
    obsPts = [10 12 14 16 26 28 30 32 42 44 46 48 58 60 62 64];
    I = speye(N^2);
    A = I(obsPts,:);
     
    [K1,K2] = meshgrid(0:N-1,0:N-1);
    rho = sqrt(prior.kappa)*(K1.^2+K2.^2).^(-prior.s);
    rho(1) = 0;
    build2 = @(U) N*idct2(reshape(U,N,N).*rho);
    build = @(U) N*reshape(idct2(reshape(U,N,N).*rho),N^2,1);
    
    %% Define the potential
    Phi = @(xi) norm((A*build(T(xi)) - data.noisy)./data.gamma)^2/2;
    DPhi = @(xi) DT(xi).*dct2((A'*(A*build(T(xi))-data.noisy))).*rho(:)/data.gamma^2./1;
    
    Xi1= @(q,v,t) [q;v-t*DPhi(q)];
    Xi2 = @(q,v,t) [q*cos(t) + v*sin(t);-q*sin(t) + v*cos(t)];

    
    for jj=1:length(betaSet)

        mcmc.beta = betaSet(jj);
        h = mcmc.beta;

        fprintf('N = %i, beta = %f\n',N,mcmc.beta)
        
        %% Run the MCMC
        % Define the initial MCMC state
        xi = normrnd(0,1,N^2,1);
        xiTT = reshape(xiT,NT,NT);
        xi = reshape(xiTT(1:N,1:N),N^2,1);

        %phiXi = Phi(xi);

        % Track the acceptance rates for the proposals
        mcmc.accepted.field = zeros(mcmc.samples,1);

        % Store the traces of tau and the lowest frequency modes
        traceXi = zeros(mcmc.samples,8);
        xiMean = xi;

        % Perform the MCMC
        for s=1:mcmc.samples
            traceXi(s,:) = [xi(2),xi(3),xi(N+1),xi(N+2),xi(N+3),xi(2*N+1),xi(2*N+2),xi(2*N+3)];

            if s == 1
                display_figures;
                if output.save_figures == 1
                    fname = sprintf('data/step_0.png');
                    print('-r144','-dpng',fname);
                end
            end

            % Provide output every output.thinning samples
            if mod(s,output.thinning) == 0
                fprintf('s = %i\n',s);
                fprintf('Acceptance rate for xi:\t\t%f\n',mean(mcmc.accepted.field(s-output.thinning+1:s)));

                ind = s/output.thinning;
                xiMean = (ind*xiMean + xi)./(ind+1);

                if output.figures == 1
                    display_figures;
                end

                if output.save_figures == 1
                    fname = sprintf('data/step_%i.png',int64(s/output.thinning));
                    print('-r144','-dpng',fname);
                end
            end

            %% Update xi | y

            % Generate the pCN proposal
            q = xi;
            v = normrnd(0,1,N^2,1);
            
            fq = -DPhi(q);
            DH = -Phi(q) + h^2/8*norm(fq)^2/N^2 + h/2*fq'*v/N^2;
            
            for tt = 1:floor(TMax/h)-1
                temp = Xi1(q,v,h/2);
                q = temp(1:N^2); v = temp(N^2+1:end);
                temp = Xi2(q,v,h);
                q = temp(1:N^2); v = temp(N^2+1:end);
                temp = Xi1(q,v,h/2);
                q = temp(1:N^2); v = temp(N^2+1:end);
                
                fq = -DPhi(q);
                DH = DH + h*fq'*v/N^2;
                
                %surf(reshape(build(T(q)),N,N));
                %pause(0.01);
            end
            
            temp = Xi1(q,v,h/2);
            q = temp(1:N^2); v = temp(N^2+1:end);
            temp = Xi2(q,v,h);
            q = temp(1:N^2); v = temp(N^2+1:end);
            temp = Xi1(q,v,h/2);
            q = temp(1:N^2); v = temp(N^2+1:end);
           
            fq = -DPhi(q);
            DH = DH + Phi(q) - h^2/8*norm(fq)^2/N^2 + h/2*fq'*v/N^2;
            
            xiProp = q;

            % Calculate the acceptance probability aProb
            %phiXiProp = Phi(xiProp);
            aProb = min(1,exp(-DH));

            % Accept proposal with probability aProb
            if unifrnd(0,1) < aProb
                xi = xiProp;
                %phiXi = phiXiProp;
                %mcmc.accepted.field(s) = 1;
            end
            mcmc.accepted.field(s) = aProb;
            
        end

        acceptMean(jj,kk) = mean(mcmc.accepted.field);

    end
end


































