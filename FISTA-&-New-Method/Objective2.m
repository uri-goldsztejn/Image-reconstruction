function [ obj, dMisfit, reg1Misfit ,reg2Misfit ] = Objective2( A,m,d,gamma1,gamma2,dtheta )
% Returns the objective and the misfit terms

dMisfit = norm(A*m-d)^2;

m_square = reshape(m,256,256);
reg1Misfit = sum(sum(sqrt(Dx(m_square).^2+Dy(m_square).^2 )));

if nargin == 4
    obj = dMisfit + gamma1*reg1Misfit;
    reg2Misfit = 0;
    return;
else

lengthImg = 256;
x0 = lengthImg/2;
y0 = lengthImg/2;
dtheta_rad = dtheta*pi/180;
m_square = reshape(m,256,256);
partialX = Dy(m_square);
partialY = Dx(m_square);
D_grad_m = zeros(lengthImg,lengthImg,2);
mask = zeros(lengthImg,lengthImg);
grad_m_D_grad_m = zeros(lengthImg,lengthImg);

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
        grad_m_D_grad_m(x,y) = [partialX(x,y),partialY(x,y)]*squeeze(D_grad_m(x,y,:));
        mask(x,y) = min(1, sqrt((x-x0)^2+(y-y0)^2)/lengthImg);
    end
end
        [row, col] = find(isnan(grad_m_D_grad_m(:,:)));
        grad_m_D_grad_m(row,col) = 0;
        cost_pixel = mask.*grad_m_D_grad_m;
        
reg2Misfit = sum(sum(cost_pixel));
obj = dMisfit + gamma1*reg1Misfit + gamma2*reg2Misfit;

end

end

