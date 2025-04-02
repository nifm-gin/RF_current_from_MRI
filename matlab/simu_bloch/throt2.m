function varargout=throt2(phi,theta)
% function Rth=throt2(phi,theta)
% function Rth=throt2(rot)
% function [Rthx,Rthy,Rthz] = throt2(...)
%
% Calculate the matrix representing a 3D rotation by an angle phi around an
% axis in the x-y plane forming an angle of theta with respect to the x
% axis. Angles are in units of radians.
% Alternatively, input complex vector of rotations, so that phi=abs(rot)
% and theta=angle(rot). In the latter case, calculations are faster and
% very slightly more precise.
% The '2' in the name of the function indicates that the N rotation
% matrices are stacked in the 2nd dimension: The output is of size 3xNx3,
% where N is the number of elements in phi and theta. The first dimension
% are the rows of the matrix, the last one are the columns. This is done so
% that Rth(:,:,i) represents rows that can be multiplied point-by-point
% with a 3xN matrix of magnetization vectors for fast matrix multiplication
% for all N elements in parallel:
%    for k=1:3
%        M_new(k,:) = sum(Rth(:,:,k).*M_old);
%    end
%
% If called with three output arguments, returns the columns in separate
% variables such that:
%    M_new = [sum(Rthx.*M_old);sum(Rthy.*M_old);sum(Rthz.*M_old)];
% 
% based on code from http://www-mrsrl.stanford.edu/~brian/bloch/
%
% extensive modifications by Jan Warnking
% Univ. Grenoble Alpes, Grenoble Institut des Neurosciences, GIN, F-38000 Grenoble, France
% Inserm, U1216, F-38000 Grenoble, France
%

switch nargin
    case 1
        rot = phi;
        
        N = numel(rot);
        if N > 1
            rot = reshape(rot,[1,N,1]);
        end
        
        phi = abs(rot);
        rot = rot./phi;  % normalize to a 
        rot(phi==0) = 0; % avoid NaNs in the result

        ct = real(rot);  % cosinus theta
        st = imag(rot);  % sinus   theta
        
    case 2
        n_phi = numel(phi);
        n_the = numel(theta);
        N = max(n_phi,n_the);
        
        % if either of phi and theta is scalar, all is fine
        if ((n_the ~= N) && (n_the ~= 1)) || ((n_phi ~= N) && (n_phi ~= 1))
            error('Theta and phi must have the same number of elements or at least of them one must be scalar.');
        end
        
        ct = cos(theta);
        st = sin(theta);
        if n_the > 1
            ct  = reshape(ct,[1,N,1]);
            st  = reshape(st,[1,N,1]);
        end
        if n_phi > 1
            phi = reshape(phi,[1,N,1]);
        end
        
        
    otherwise
        error('There must be one or two input arguments');
end

% we should calculate:
% Rz = zrot(-theta);
% Rx = xrot(phi);
% Rth = Rz\Rx*Rz;
% 
% But we can't do matrix multiply since Rz and Rx are potentially 3x3xN.
% In this case manual calculation is faster anyway.

sp = sin(phi);
cp = cos(phi);
ct2 = ct.*ct;
st2 = 1-ct2;                    % sin(theta).^2 = 1 - cos(theta).^2

Rx = zeros(3,N);
Ry = Rx;
Rz = Rx;

Rx(1,:) = ct2+st2.*cp;
Rx(2,:) = (ct.*st).*(1-cp);
Rx(3,:) = st.*sp;
Ry(1,:) = Rx(2,:);
Ry(2,:) = st2+ct2.*cp;
Ry(3,:) =-ct.*sp;
Rz(1,:) =-st.*sp;
Rz(2,:) = ct.*sp;
Rz(3,:) = cp;

switch max(nargout,1)
    case 1
        varargout{1} = reshape([Rx Ry Rz],[3,N,3]);
    case 3
        varargout{1} = Rx;
        varargout{2} = Ry;
        varargout{3} = Rz;
end
