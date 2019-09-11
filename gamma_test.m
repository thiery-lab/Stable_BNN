M = 100;
g = 101;

x = unifrnd(0,1,M,1);
y = unifrnd(0,1,M,1);
u = gamrnd(1,2,M,1);
xx = linspace(0,1,g);
yy = linspace(0,1,g);
zz = zeros(g,g);

for i=2:g
    xok = (x<xx(i));
    for j=2:g
        yok = (y<yy(j));
        zz(i,j) = sum(u(xok&yok));
    end
end

surf(zz,'EdgeColor','None');