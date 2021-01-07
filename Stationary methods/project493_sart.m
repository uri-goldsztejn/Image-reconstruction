%% stationary algorithm 2 SART
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
% Implement SART 
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
disp('>>>> Implementing SART');
data_res = size(A,1); % size of data
img_res = size(A,2); % size of m
% diagonal matrix where each diagonal element is the sum of all row
% entries in the same column
row_sum = sparse(img_res,img_res);
for i = 1:img_res
    row_sum(i,i)=1/norm(A(:,i),1);
end
% diagonal matrix where each diagonal element is the sum of all column
% entries in the same row
col_sum = sparse(data_res,data_res);
for j = 1:data_res
    col_sum(j,j)=1/norm(A(j,:),1);
end
AT_normalized = row_sum*A'*col_sum;

% % % % 
max_iter = 300;
% stores the residual history, the 1st entry is norm of data
residual_history = zeros(max_iter+1,1);
% stores the history of the norm m^k, 1st entry is 0
m_norm_history = zeros(max_iter+1,1);
% initial guess of m is a zero vector
m_k = zeros(img_res,1);
% difference in norm(m^k) from the last 2 iterations, default is greater
% than tol until modified
dm_k = 10;
% store dm_k history
dm_history = zeros(max_iter+1,1);
% set the toleration for dm_k
tol = 6;
i=1;

while (dm_k>tol)&&(i<max_iter)
    
    m_norm_history(i) = norm(m_k);
    residual_k = data - A * m_k;
    residual_history(i)=norm(residual_k);
    
    step_k = AT_normalized * residual_k;
    m_k_next = m_k + step_k;
    dm_history(i) = norm(step_k);
    dm_k = norm(m_k_next - m_k);
    
    m_k = m_k_next;
    disp(sprintf('>>>>Iteration:%d',i));
    i = i+1;
end
residual_history(i)=norm(data-A*m_k);
m_norm_history(i)=norm(m_k);

% % find the number of iterations by locating the last nonzero entry in the
% % residual history
% iter = find(residual_history,1,'last');
iterations = 1:i;
% plot the convergence history of residual and ||m^(k)||
figure
subplot(2,1,1)
plot(iterations,m_norm_history(1:i),'Linewidth',3)
title('SART convergence history','fontsize',14)
ylabel('||m^k||','fontsize',13,'fontweight','bold')

subplot(2,1,2)
plot(iterations,residual_history(1:i),'Linewidth',3)
xlabel('Iterations','fontsize',13,'fontweight','bold')
ylabel('Residual','fontsize',13,'fontweight','bold')

figure;
imagesc(reshape(m_k,[256,256]));