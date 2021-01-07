%% stationary algorithm 1 ART
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Load Files, Initialize workspace
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
clear; clc; close all;
load('project_data.mat');
m_true= imgref;

% compressed sampling factor
compress = 6;
data = zeros(size(sinogram,1),size(sinogram,2)/compress);
for i=1:(size(sinogram,2)/compress)
    data(:,i)=sinogram(:,compress*i);
end

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
nviews= 540/compress;

%Angle increment between views (unit:degree)
dtheta= compress*5/12;
%Views
views= [0:nviews-1]*dtheta;

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

data= reshape(data, [nviews*nrays, 1]);

% Remove possibly 0 rows from K and d
[A, data] = purge_rows(A, data);

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Implement ART
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
Iterations=1:1:100;
max_iter = 100;
% initial guess of m is a zero vector
m_updates_ART_NN= zeros(size(A,2),1);

m_Norm_history = zeros(size(Iterations));
Residual_ART_NN= zeros(size(Iterations));
step_norm_history = zeros(size(Iterations));
Mean_Square_Error_ART_NN= zeros(size(Iterations));

norm_A_rows = sum(A.*A, 2);
AT = A';
m_true_vector = reshape(m_true,[npixels*npixels,1]);
% difference in norm(m^k) from the last 2 iterations, the initial value is
% set to be greater than tolerance
dm_k = 100;
% set the toleration for dm_k
tol = 50;
i=1;

while (dm_k>tol)&&(i<max_iter)
    % make a copy of m_k from the last iteration
    m_k_previous = m_updates_ART_NN;
    
    for j=1:1:size(A,1)
    
     Descent = (data(j) - ( m_updates_ART_NN'*AT(:,j)))/norm_A_rows(j);
     m_updates_ART_NN = m_updates_ART_NN + Descent*AT(:,j);    
        
    end
    
    m_Norm_history(i) = norm(m_updates_ART_NN);
    step_norm_history(i) = norm(m_updates_ART_NN-m_k_previous);
    dm_k = step_norm_history(i);
   
    Residual_ART_NN(i)= norm(data-A*m_updates_ART_NN);
    Mean_Square_Error_ART_NN(i)= norm(m_true_vector - m_updates_ART_NN);
    disp(sprintf('>>>>At iteration:%d',i));
    i = i+1;
end

figure
imagesc(reshape(m_updates_ART_NN,[npixels,npixels]))

figure
plot(Iterations, Residual_ART_NN)
% find the number of iterations by locating the last nonzero entry in the
% residual history
iter = find(Residual_ART_NN,1,'last');

Residual_ART_NN = [norm(data),Residual_ART_NN(1:iter)];
m_Norm_history = [0, m_Norm_history(1:iter)];
Iterations = [0 Iterations(1:iter)];

% plot the convergence history of residual and ||m^(k)||
figure;
subplot(2,1,2)
plot(Iterations, m_Norm_history,'Linewidth',3);
xlim([0,60])
xlabel('Iterations','fontsize',13,'fontweight','bold')
ylabel('||m^k||','fontsize',13,'fontweight','bold')
subplot(2,1,1)
plot(Iterations,Residual_ART_NN,'Linewidth',3);
title('ART convergence history (540 Views)','fontsize',14)
xlim([0,60])
ylabel('Residual','fontsize',13,'fontweight','bold')