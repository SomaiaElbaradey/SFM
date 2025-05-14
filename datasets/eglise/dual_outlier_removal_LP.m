load ../../../reconstruct_scene_sparse/eglise/impoints4.mat
load ../../../reconstruct_scene_sparse/eglise/rotations3.mat
load ../../uwo/kalib/Calib_Results.mat KK;

u2 = u;
u = cell(1,length(A));
for i = 1:length(u);
    u{i} = NaN(3,u2.pointnr);
    u{i}(:,u2.index{i}) = u2.points{i};
end
clear u2;
cams = 1:length(u);

A = A(cams);
u = u(cams);
pixtol = 5;

vis = zeros(1,size(u{1},2));
for i=1:length(u)
    vis = vis + isfinite(u{i}(1,:));
end
vis = vis >= 2;
for i = 1:length(u)
    u{i} = u{i}(:,vis);
end

tic
%known rotation
[U,P,dual,s] = krot_feas_LP(u,A,pixtol/KK(1,1),0.1,100);
outlseq = [];
iter = 1;
while s > 0
    %Ta bort outliers
    Uold = U;
    Pold = P;
    outlnr = 0;
    figure(1);
    plot(full(dual));
    drawnow;
    res2 = [];
    for i = 1:length(u);
        vis = find(isfinite(u{i}(1,:)));
        res2 = [res2 max(abs(u{i}(:,vis)-pflat(P{i}*U(:,vis))))];
        res = length(vis);
        outl = find(dual(1:res) > 1e-5);
        outlnr = outlnr + sum(dual(1:res) > 1e-5);
        u{i}(:,vis(outl)) = NaN;
        dual = dual(res+1:end);
    end
    figure(2);
    plot(res2);
    drawnow;
    outlseq = [outlseq outlnr];
    vis = zeros(1,size(u{1},2));
    for i=1:length(u)
        vis = vis + isfinite(u{i}(1,:));
    end
    vis = vis >= 2;
    if vis(1) == 0;
        disp('första punkten outlier');
    end
    for i = 1:length(u)
        u{i} = u{i}(:,vis);
    end
    [U,P,dual,s] = krot_feas_LP(u,A,pixtol/KK(1,1),0.1,100);
    iter = iter+1;
end
t = toc;

P = Pold;
U = Uold(:,vis);

plot(motion(P),structure(U));
h = gcf;
h = get(h,'children');
child = get(h,'children');
for i = 1:length(child);
    if length(get(child(i),'XData')) == size(U,2);
        set(child(i),'Marker','.');
        set(child(i),'MarkerSize',1);
    end
end
%axis([-.3 .3 -.3 .3 0 .5])

imseq = cell(0);
for i = 1:length(u);
    imseq{i} = imagedata([],u{i});
end

[s,m] = modbundle(structure(U),motion(P),imseq,20,0.1);
figure;
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
%axis([-.3 .3 -.3 .3 0 .5])

figure;
reproject(s,m,imseq);


if 0; %Med färger
    U = pflat(getpoints(s));
    plot(m);
    hold on;
    for i = 1:size(U,2);
        plot3(U(1,i),U(2,i),U(3,i),'Color',RGB(:,i)'/255,'Marker','.', 'MarkerSize',3);
    end
    hold off;
    axis([-.3 .3 -.3 .3 0 .5])
end

if 0 %exportera till visualize
    %filnamn
    imfile = cell(0);
    imnames = dir('*.JPG');
    imnames = imnames(cams);
    for i = 1:length(imnames);
        imfile{end+1} = imnames(i).name;
    end
    
    %okalib kamror
    P = cell(0);
    imseq2 = cell(size(imseq));
    for i = 1:size(m);
        P{i} = KK*getcameras(m,i);
        imseq2{i} = imagedata(imfile{i},pflat(KK*getpoints(imseq{i})));
        
    end
    m_uncalib = motion(P);
    
    %rgbvärde
    RGB = zeros(3,size(s,1));
    nr = zeros(1,size(s,1));
    for i = 1:size(m);
        i
        u = getpoints(imseq{i});
        vis = isfinite(u(1,:));
        p = pflat(KK*u(:,vis));
        im = imread(imfile{i});
        RGB(:,vis) = RGB(:,vis)+impixel(im,p(1,:),p(2,:))';
        nr(:,vis) = nr(:,vis)+1;
    end
    RGB = RGB./repmat(nr,[3 1]);
    RGB = uint8(RGB);
    visualize_export('Hus.out',s,m_uncalib,imfile,RGB);
end

if 0
    save Hus.mat s m imseq m_uncalib imseq2;
end
