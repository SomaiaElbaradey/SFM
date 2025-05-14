function [U,P,dual,s] = krot_feas_LP(u,A,tol,min_depth,max_depth)

numcams = length(A);
[a,a0,b,b0,c,c0,Aeq,Beq] = gen_krot(u,A);
[Linfsol,dual,s] = LinfSolverfeas(a,a0,b,b0,c,c0,Aeq,Beq,tol,min_depth,max_depth);
[U,P] = form_str_mot(u,A,Linfsol);


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function [U,P] = form_str_mot(u,A,sol)
numpts = size(u{1},2);
numcams = length(A);

U = reshape(sol(1:(3*numpts)),[3 numpts]);
U = pextend(U);

%tpart = [sol(3*numpts+1:end)];
%tpart = [sol(3*numpts+1:end); -1; 0; 0; 0];
%tpart = [0;0;0;1;sol(3*numpts+1:end)];
tpart = [0;0;0;sol(3*numpts+1:end)];
P = cell(size(A));
for i=1:length(A)
    P{i} = [A{i} tpart((i-1)*3+1:i*3)];
end


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function [Y,dual,s] = LinfSolverfeas(a,a0,b,b0,c,c0,Aeq,Beq,tol,min_depth,max_depth)
	
if 0 %Linprog variant
    A1 = [-a-tol*c; a-tol*c; -b-tol*c; b-tol*c];
    B1 = [a0+tol*c0; -a0+tol*c0; b0+tol*c0; -b0+tol*c0];
    % Djupbegränsningar
    A2 = [-c; c];
    B2 = [c0-min_depth; max_depth-c0];
    
    A = [A1 -ones(size(A1,1),1);A2 zeros(size(A2,1),1)];
    C = [B1;B2];
    B = [sparse(size(A1,2),1); 1];
    %opts = optimset('linprog');
    %opts.MaxIter = 5000;
    %[X,Fval,flag,output,dual] = linprog(C,A,B,[],[],[],[],[],opts);
    K.l = size(A,1);
    pars.eps = 1e-8;
    [X,Y,info] = sedumi(A,-B,C,K,pars);
    s = Y(end);
    Y = Y(1:end-1);
    res = size(a,1);
    dual = X(1:res)+X(res+1:2*res)+X(2*res+1:3*res)+X(3*res+1:4*res)+X(4*res+1:5*res)+X(5*res+1:end);
end
if 1 %Mosek linprog variant
    A1 = [-a-tol*c; a-tol*c; -b-tol*c; b-tol*c];
    B1 = [a0+tol*c0; -a0+tol*c0; b0+tol*c0; -b0+tol*c0];
    % Djupbegränsningar
    A2 = [-c; c];
    B2 = [c0-min_depth; max_depth-c0];
    
    A = [A1 -ones(size(A1,1),1);A2 zeros(size(A2,1),1)];
    buc = [B1;B2];
    C = [sparse(size(A1,2),1); 1];
    par = msklpopt(C,A,[],buc,[],[],[],'param');
    par.param.MSK_IPAR_INTPNT_BASIS = 'MSK_BI_NEVER'; %Stäng av simplex lösare
    par.param.MSK_DPAR_INTPNT_CO_TOL_MU_RED = 1e-12;
    result = msklpopt(C,A,[],buc,[],[],par.param,'minimize echo(0)');
    
    s = result.sol.itr.pobjval
    Y = result.sol.itr.xx(1:end-1);
    X = result.sol.itr.suc;
    res = size(a,1);
    dual = X(1:res)+X(res+1:2*res)+X(2*res+1:3*res)+X(3*res+1:4*res)+X(4*res+1:5*res)+X(5*res+1:end);
end

    

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function [a,a0,b,b0,c,c0,Aeq,Beq] = gen_krot(u,A)
numvar = 3*size(u{1},2)+3*length(A);
numpts = size(u{1},2);
numcams = length(A);
%StÃ¤ll upp problemet. 
a = [];
b = [];
c = [];
for i = 1:numcams;
    R = A{i};
    p = u{i};
    visible_points = isfinite(p(1,:));
    numres = sum(visible_points);
    
    %FÃ¶rsta termen
    %BerÃ¤kna koefficienter framfÃ¶r 3D-punkten i varje residual
    ptind = find(visible_points');
    pointcoeff = p(1,visible_points)'*R(3,:)-ones(numres,1)*R(1,:);
    %rad och kollon i a-matrisen
    pointcol = [(ptind-1)*3+1 (ptind-1)*3+2 ptind*3];
    pointrow = [1:numres]'*[1 1 1];
    
    %Koeff. framfÃ¶r translationsbiten i kameran
    tcoeff = [-ones(numres,1) zeros(numres,1) p(1,visible_points)'];
    tcol = ones(numres,1)*[numpts*3+[(i-1)*3+1:i*3]];
    trow = pointrow;
    
    %Skapa ny a matris och fyll i data
    data = [pointcoeff(:); tcoeff(:)];
    row = [pointrow(:);  trow(:)];
    col = [pointcol(:); tcol(:)];
    newa = sparse(row,col,data,numres,numvar);

    
    %Andra termen
    %BerÃ¤kna koefficienter framfÃ¶r 3D-punkten i varje residual
    ptind = find(visible_points');
    pointcoeff = p(2,visible_points)'*R(3,:)-ones(numres,1)*R(2,:);
    %rad och kollon i b-matrisen
    pointcol = [(ptind-1)*3+1 (ptind-1)*3+2 ptind*3];
    pointrow = [1:numres]'*[1 1 1];
    
    %Koeff. framfÃ¶r translationsbiten i kameran
    tcoeff = [zeros(numres,1) -ones(numres,1) p(2,visible_points)'];
    tcol = ones(numres,1)*[numpts*3+[(i-1)*3+1:i*3]];
    trow = pointrow;
    
    %Skapa ny b matris och fyll i data
    data = [pointcoeff(:); tcoeff(:)];
    row = [pointrow(:);  trow(:)];
    col = [pointcol(:); tcol(:)];
    newb = sparse(row,col,data,numres,numvar);

    %NÃ¤mnaren
    %BerÃ¤kna koefficienter framfÃ¶r 3D-punkten i varje residual
    ptind = find(visible_points');
    pointcoeff = ones(numres,1)*R(3,:);
    %rad och kollon i b-matrisen
    pointcol = [(ptind-1)*3+1 (ptind-1)*3+2 ptind*3];
    pointrow = [1:numres]'*[1 1 1];
    
    %Koeff. framfÃ¶r translationsbiten i kameran
    tcoeff = [zeros(numres,1) zeros(numres,1) ones(numres,1)];
    tcol = ones(numres,1)*[numpts*3+[(i-1)*3+1:i*3]];
    trow = pointrow;
    
    %Skapa ny b matris och fyll i data
    data = [pointcoeff(:); tcoeff(:)];
    row = [pointrow(:);  trow(:)];
    col = [pointcol(:); tcol(:)];
    newc = sparse(row,col,data,numres,numvar);

    if 0;
        UU = rand(3,numpts);
        UUU = UU(:);
        t = rand(3, numcams);
        tt = t(:);
        var = [UUU; tt];
        slask = R*UU + repmat(t(:,i),[1 size(UU,2)]);
        figure(1);plot((p(1,visible_points).*slask(3,visible_points)-slask(1,visible_points))'-newa*var)
        figure(2);plot((p(2,visible_points).*slask(3,visible_points)-slask(2,visible_points))'-newb*var)
        figure(3);plot(slask(3,visible_points)'-newc*var)
    end

    %LÃ¤gg in i matriserna
    a = [a;newa];
    b = [b;newb];
    c = [c;newc];
end


% a = a(:,[1:numpts*3 (numpts*3+4):end]);
% b = b(:,[1:numpts*3 (numpts*3+4):end]);
% c = c(:,[1:numpts*3 (numpts*3+4):end]);
a0 = zeros(size(a,1),1);
b0 = zeros(size(b,1),1);
c0 = zeros(size(c,1),1);

%Välj koordinatsystem så att 
%första kameran = [A [0 0 0]'];
a = a(:,[1:3*numpts 3*numpts+4:end]);
b = b(:,[1:3*numpts 3*numpts+4:end]);
c = c(:,[1:3*numpts 3*numpts+4:end]);


%Välj koordinatsystem så att 
%första kameran = [A [0 0 0]'];
%andra kameran [A [1 * *]'];
%a0 = a(:,3*numpts+4);
%b0 = b(:,3*numpts+4);
%c0 = c(:,3*numpts+4);
%a = a(:,[1:3*numpts 3*numpts+5:end]);
%b = b(:,[1:3*numpts 3*numpts+5:end]);
%c = c(:,[1:3*numpts 3*numpts+5:end]);

%Välj koordinatsystem så att 
%sista kameran = [A [0 0 0]'];
%näst sista kameran [A [* * -1]'];
% a = a(:,1:end-3);
% b = b(:,1:end-3);
% c = c(:,1:end-3);
% a0 = -a(:,end);
% b0 = -b(:,end);
% c0 = -c(:,end);
% a = a(:,1:end-1);
% b = b(:,1:end-1);
% c = c(:,1:end-1);

%Koordinatsystem så att medelpunkten av kamrorna ligger i nollan
Aeq = zeros(3,numpts*3);
for i = 1:length(A)
    Aeq = [Aeq -inv(A{i})];
end
Beq = zeros(3,1);


if 0;
    R = A{i};
    UU = rand(3,numpts);
    UU(:,1) = 0;
    UUU = UU(:);
    t = rand(3, numcams);
    t(3,1) = 1;
    tt = t(:);
    var = [UUU; tt];
    slask = R*UU + repmat(t(:,i),[1 size(UU,2)]);
    figure(1);plot((p(1,visible_points).*slask(3,visible_points)-slask(1,visible_points))'-newa*var)
    figure(2);plot((p(2,visible_points).*slask(3,visible_points)-slask(2,visible_points))'-newb*var)
    figure(3);plot(slask(3,visible_points)'-newc*var)
end
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%