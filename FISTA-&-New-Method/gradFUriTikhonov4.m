function [grad] = gradFUriTikhonov3(m_k,dtheta )
%UNTITLED2 Summary of this function goes here
%   Detailed explanation goes here
% Rd = zeros(2,2);
lengthImg = 256;
x0 = lengthImg/2;
y0 = lengthImg/2;
dtheta_rad = dtheta*pi/180;
m_square = reshape(m_k,256,256);
partialX = Dy(m_square);
partialY = Dx(m_square);
D_grad_m = zeros(lengthImg,lengthImg,2);
mask = zeros(lengthImg,lengthImg);

for x = 1:lengthImg
    for y =1:lengthImg

        %define angle at specified location
        alpha = atan2((y - y0),(x-x0));
        if alpha<0
            alpha = alpha + pi;
        end
        %round
        if mod(alpha,dtheta_rad) < dtheta_rad/2
            alpha_round = floor(alpha/dtheta_rad)*dtheta_rad;
        else
            alpha_round = floor(alpha/dtheta_rad)*dtheta_rad + dtheta_rad;
        end
        
        %calculate D*grad(m)
        di = [-sin(alpha_round),cos(alpha_round)];
%         di = [cos(alpha_round),sin(alpha_round)];
        D = di'*di;
        D_grad_m(x,y,:) = D*[partialX(x,y);partialY(x,y)];
        
        mask(x,y) = min(1, sqrt((x-x0)^2+(y-y0)^2)/lengthImg);
    end
end
        [row, col] = find(isnan(D_grad_m(:,:,1)));
        D_grad_m(row,col,:) = 0;
        
        % gradient is then -div(D*grad(m))
        
       Fx = -Dyt(D_grad_m(:,:,1));
       Fy = -Dxt(D_grad_m(:,:,2));
        
       divGrad = - mask.*(Fx + Fy);
       grad = reshape(divGrad,[],1);


end

