function [ac,fc] = LineSearch(x,d,a0,ni,OptFn);
% [ac,fc] = LineSearch(x,d,a0,ni);
%
%   x    Starting point in the multivariate space
%   d    Search direction
%   a0   Initial guess about the line search step size
%   ni   Maximum number of iterations (queries to OptFn)
%   
%   ac   Step size found
%   fc   Value of the function evaluated at (x + ac*d)
%
%   Implementation of quadratic line search algorithm with
%   interval bounding and a safeguard based on the bisection
%   algorithm. Designed to be used as part of a multivariate
%   optimization algorithm where the function being minimized
%   is called "OptFn" and has the form f = OptFn(x). If no
%   outputs are specified, this displays a plot of the 
%   optimization function and the query points.

%clear all;
%close all;

c = 2; % Interval bound expansion rate

a = zeros(ni,1);

al = -max(-a0,0);  fl = OptFn(x+d*al); a(1) = al; % Left edge
ar = abs(a0);      fr = OptFn(x+d*ar); a(3) = ar; % Right edge
ac = (ar + al)/2;  fc = OptFn(x+d*ac); a(2) = ac; % Center
cnt = 3;

while cnt<ni,
    if al>=0,
        if fc>=fr,
            al = ac;   fl = fc;
            ac = ar;   fc = fr;
            ar = ar*c; fr = OptFn(x+d*ar);
            cnt    = cnt + 1;
            a(cnt) = ar;
        elseif fc>=fl,
            ar = ac;    fr = fc;
            ac = al;    fc = fl;
            al = al/10; fl = OptFn(x+d*ar);
            cnt    = cnt + 1;
            a(cnt) = al;        
        else  % fc<fr & fc<fl,
            break;
            end;
        
    else
        if fc>=fr,
            ac = ar;   fc = fr;
            ar = ar*c; fr = OptFn(x+d*ar);
            cnt    = cnt + 1;
            a(cnt) = ar;  
        elseif fc>=fl
            ac = al;   fc = fr;
            al = al*c; fl = OptFn(x+d*al);
            cnt    = cnt + 1;
            a(cnt) = al;  
        else
            break;
            end;
        end;    
    end;
       
while cnt<ni,
    if fc<fl & fc<fr, % Criteria okay - do quadratic search iteration
        acr = ac - ar; bcr = ac^2 - ar^2;
        arl = ar - al; brl = ar^2 - al^2;
        alc = al - ac; blc = al^2 - ac^2;
        
        num = (bcr*fl + brl*fc + blc*fr);
        den = (acr*fl + arl*fc + alc*fr);
        if den==0, % Optimal solution found
            break;
            end;
        an = 0.5*num/den;
        fn = OptFn(x+d*an);
        cnt = cnt + 1;
        a(cnt) = an;
        if an>ac,
            if fn>=fc,
                ar = an; fr = fn;
            else
                al = ac; fl = fc;
                ac = an; fc = fn;
                end;
        else
            if fn>=fc,
                al = an; fl = fn;
            else
                ar = ac; fr = fc;
                ac = an; fc = fn;
                end;   
            end;
    else % Safety switchover to bisection - safeguard technique
        a2  = ac + 0.001*(ar-al);
        f2  = OptFn(x+d*a2);
        cnt    = cnt + 1;
        a(cnt) = a2;
        if fc<f2,
            ar = a2; fr = f2;
        else % fc>fr
            al = ac; fl = fc;
            end;
        ac     = (al+ar)/2; 
        fc     = OptFn(x+d*ac);
        cnt    = cnt + 1;        
        a(cnt) = ac;
        end;
    if ar-al==0 | (fl==fr & fl==fc),
        break;
        end;
    end;
           
if nargout==0,
    figure;
    FigureSet(1,4.5,2.7);
    u  = min(a):(max(a)-min(a))/1e3:max(a);
    fu = zeros(size(u));
    for c1 = 1:length(u),
        fu(c1) = OptFn(x+d*u(c1));
        end;
    fa = zeros(size(a));
    for c1 = 1:length(a),
        fa(c1) = OptFn(x+d*a(c1));
        end;    
    h = plot(u,fu,'r',a,fa,'.');
    set(h(2),'MarkerSize',8);
    set(h,'LineWidth',1.2);
    ylabel('Function');
    xlabel('Input');
    box off;
    AxisSet(8);
    fprintf('Pausing...\n'); pause;
    end;
