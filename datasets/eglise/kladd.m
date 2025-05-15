
U = getpoints(s);
C = mean(pflat(U),2);
sd = sqrt(sum(sum((U-repmat(C,[1 size(U,2)])).^2)/size(U,2)));

plot(m,s);
h = gcf;
h = get(h,'children');
child = get(h,'children');
for i = 1:length(child);
    if length(get(child(i),'XData')) == size(s,1);
        set(child(i),'Marker','.');
        set(child(i),'MarkerSize',1);
    end
end
axis(2*sd*[-1 1 -1 1 -1 1]+C([1 1 2 2 3 3])')
xlabel('x');
ylabel('y');
zlabel('z');

vis = 0;
for i = 1:length(imseq2);
    u = getpoints(imseq2{i});
    vis = vis + isfinite(u(1,:));
end

figure(2);
s2 = structure(U(:,vis > 3));
plot(m,s2);
h = gcf;
h = get(h,'children');
child = get(h,'children');
for i = 1:length(child);
    if length(get(child(i),'XData')) == size(s2,1);
        set(child(i),'Marker','.');
        set(child(i),'MarkerSize',1);
    end
end
hold on;
plot(structure(U(:, vis < 3)),'g.');
axis(2*sd*[-1 1 -1 1 -1 1]+C([1 1 2 2 3 3])')



figure(3);
U2 = pflat(U(:,vis > 3));
for i = 1:length(imseq);
    u = getpoints(imseq{i});
    imseq2{i} = imagedata(getfilename(imseq{i}),u(:,vis > 3));
end
plot(m_uncalib,structure(U2));
U = getpoints(s);
C = mean(pflat(U),2);
sd = sqrt(sum(sum((U-repmat(C,[1 size(U,2)])).^2)/size(U,2)));

h = gcf;
h = get(h,'children');
child = get(h,'children');
for i = 1:length(child);
    if length(get(child(i),'XData')) == size(structure(U2),1);
        set(child(i),'Marker','.');
        set(child(i),'MarkerSize',1);
    end
end
axis(2*sd*[-1 1 -1 1 -1 1]+C([1 1 2 2 3 3])')
xlabel('x');
ylabel('y');
zlabel('z');
