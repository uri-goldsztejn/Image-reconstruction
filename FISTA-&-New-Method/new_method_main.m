%% Part (B)
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Load Files, Initialize workspace
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
clear; clc; close all;
load('project_data.mat');
m_true= imgref;

data= sinogram;
data_new= sinogram;%(:,1:6:540); %Sinogram with 90 views
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%Generate Imaging Operator A
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
disp('>>>> Generating Imaging Operator');
% Size of the region of interest (unit:mm)
L= 0.06144;

% Number of pixels in each direction
npixels= 256;

% Pixel size
pixel_size= L/npixels;

%Number of view
nviews= [540,270,90];                 %Change number of views
iterations = zeros(1,3);
maxIter = 1000; %1000
costs = zeros(3,maxIter);
MSE = zeros(3,1);
middleRows = zeros(3,256);
misfits = zeros(3,maxIter);
reg1s = zeros(3,maxIter);
reg2s = zeros(3,maxIter);

for j =1:3
%Angle increment between views (unit:degree)

    fprintf('   Reconstructing image with %i views\n',nviews(j));
    
    dtheta= 225/nviews(j);             %Change angle increment between views
    data_new= sinogram(:,1:540/nviews(j):540);%(:,1:6:540); %Sinogram with 90 views
    %Views
    views= [0:nviews(j)-1]*dtheta;

    %Number of rays for each view
    nrays= 512;

    %Distance between first and last ray (unit pixels)
    d= npixels*(nrays-1)/nrays;

    % Construct imaging operator (unit:pixels)
    [A] = paralleltomo(npixels, views, nrays, d);

    %Rescale A to physical units (unit:mm)
    A= A*pixel_size;

    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    % Display Sinogram and Original Image, Reshape Sinogram and m_true
    % for Reconstruction
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    % Display reference image and sinogram
    disp('>>>> Reshaping Sinogram');
    figure
    subplot(121);
    imagesc(m_true);
    title('True image');
    subplot(122);
    imagesc(data_new);
    title('Data (sinogram)');

    data_new= reshape(data_new, [nviews(j)*nrays, 1]);

    %m_true= reshape(m_true,[size(A,2),1]);
    % figure; imagesc(reshape(data,nrays,nviews));
    % figure; imagesc(sinogram);
    % Remove possibly 0 rows from K and d
    [A, data_new] = purge_rows(A, data_new);

    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    %% Define L, eta, x_0, y_1 and t_1;

    L=1;
    x = ones(npixels^2,1);
    y = x;
    t=1;
    gamma1 =1e-4; %1e-4
    gamma2= 1e-4; %1e-4
    maxIter = 1000; %1000
    tol = 5;
    m_cs = reshape(m_true,[],1);

    %%
    % Define functions...
    % f(x)= norm(Ax-d)^2
    % g(x)= TV_I as defined on paper
    % grad_f(x) = A'(Ax-d)
    %
    % F(x)=f(x)+g(x)
    % Q(x,y,L)= f(y)+(x-y)'grad_f(y) + L/2*norm(x-y)^2+g(x)
    % Where f(x)= norm(Ax-d)^2 and g(x)= TV_I as defined on the paper
    % 
    % Define b as a function b(y,L)= y - 1/L*(grad_f(y)) 

    
        grad_f = @(x) A'*(A*x-data_new) + gamma2*gradFUriTikhonov4(x,dtheta );
        b = @(y) y-1/L*grad_f(y);
        % Write beginning of outer Loop for FISTA;
        for i = 1:maxIter
            
            
           x_next = prox_tv(reshape(b(y),npixels,npixels),gamma1);
           t_next = (1+sqrt(1+4*t^2))/2;
           x_next = reshape(x_next,[],1);
           y_next = x_next +(t-1)/t_next*(x_next-x);

           if norm(x-x_next) < tol
               break;
           end

           x = x_next;
           t = t_next;
           y = y_next;
        [costs(j,i),misfits(j,i),reg1s(j,i)] = Objective2(A,x,data_new,gamma1,gamma2,dtheta);
        fprintf('   Iter %i, obj = %e\n',i, costs(j,i));

        end
        iterations(j) = i;
        MSE(j) = norm(reshape(m_true,[],1)-x);
        x_square = reshape(x,256,256);
        middleRows(j,:) = x_square(128,:);

        fprintf('   MSE =  %e\n',MSE(j));
        %
        figure
        imagesc(reshape(x,npixels,npixels))
        tit = [num2str(nviews(j)),'views reconstruction - \gamma1 = ',num2str(gamma1),'\gamma_2 = ',num2str(gamma2)];
        title(tit);
        
    
end


% Plot Convergence History 
figure;
loglog(costs(1,:));
hold on
loglog(costs(2,:));
hold on
loglog(costs(3,:));

title('Convergence History New Method');
legend('540','270','90');

% Plot Center Rows
middleRows_True = m_true(128,:);

figure;
plot(middleRows_True);
hold on;
plot(middleRows(1,:));
title('Middle Row Intensity 540 Views Reconstruction');
legend('True Image','540 Views');
xlabel('Column');
ylabel('Intensity');

figure;
plot(middleRows_True);
hold on;
plot(middleRows(2,:));
title('Middle Row Intensity 270 Views Reconstruction');
legend('True Image','270 Views');
xlabel('Column');
ylabel('Intensity');

figure;
plot(middleRows_True);
hold on;
plot(middleRows(3,:));
title('Middle Row Intensity 90 Views Reconstruction');
legend('True Image','90 Views');
xlabel('Column');
ylabel('Intensity');
