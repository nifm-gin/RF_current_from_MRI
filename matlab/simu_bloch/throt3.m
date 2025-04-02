function Rth=throt3(varargin)
% function Rth=throt3(phi,theta)
% function Rth=throt3(rot)
%
% Calculate the matrix representing a 3D rotation by an angle phi around an
% axis in the x-y plane forming an angle of theta with respect to the x
% axis. Angles are in units of radians.
% Alternatively, input complex vector of rotations, so that phi=abs(rot)
% and theta=angle(rot). In the latter case, calculations are faster and
% very slightly more precise.
% The '3' in the name of the function indicates that the N rotation
% matrices are stacked in the 3rd dimension: The output is of size 3x3xN,
% where N is the number of elements in phi and theta.
%
% based on code from http://www-mrsrl.stanford.edu/~brian/bloch/
%
% extensive modifications by Jan Warnking
% Univ. Grenoble Alpes, Grenoble Institut des Neurosciences, GIN, F-38000 Grenoble, France
% Inserm, U1216, F-38000 Grenoble, France
%

switch nargin
    case 1
        rot = varargin{1};
        
        N = numel(rot);
        if N > 1
            rot = reshape(rot,[1,1,N]);
        end
        
        phi = abs(rot);
        rot = rot./phi;  % normalize to a 
        rot(phi==0) = 1; % avoid NaNs in the result (here: 0/0=1)

        ct = real(rot);  % cosinus theta
        st = imag(rot);  % sinus   theta
        
    case 2
        phi = varargin{1};
        theta = varargin{2};
        
        N     = numel(phi);
        n_the = numel(theta);
        if N ~= n_the
            error('phi and theta must have the same number of elements');
        end
        if N > 1
            phi   = reshape(phi,  [1,1,N]);
            theta = reshape(theta,[1,1,N]);
        end
        
        ct = cos(theta);        
        st = sin(theta);
        
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
st2 = st.*st;
stsp = st.*sp;
spct = sp.*ct;
ctst1mcp = ct.*st.*(1-cp);

Rth = [ct2+st2.*cp  ctst1mcp     stsp;...
       ctst1mcp     st2+ct2.*cp  -spct;...
       -stsp        spct         cp];



