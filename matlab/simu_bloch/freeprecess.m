function [Afp,Bfp]=freeprecess(T,T1,T2,df)
% function [Afp,Bfp]=freeprecess(T,T1,T2,df)
%
%	Simulates free precession and decay over a time interval T, given
%	relaxation times T1 and T2 and off-resonance df.  Times in seconds,
%	off-resonance in Hz.
%   Output are 3x3 matrix Afp and 3x1 vector Bfp such that the effect of
%   free precession on the magnetization vector M can be written as:
%
%     M_after = Afp * M_before + Bfp;
%
%   If T or df or both are a vectors of N durations of free-precession
%   periods and/or N offset frequencies, A will be of size 3x3xN and B of
%   size 3x1xN.
%
% based on code from http://www-mrsrl.stanford.edu/~brian/bloch/
%
% extensive modifications by Jan Warnking
% Univ. Grenoble Alpes, Grenoble Institut des Neurosciences, GIN, F-38000 Grenoble, France
% Inserm, U1216, F-38000 Grenoble, France
%

n_T = numel(T);
if n_T > 1
    T = reshape(T,[1,1,n_T]);
end

n_df = numel(df);
if n_df > 1
    df = reshape(df,[1,1,n_df]);
    if n_T>1 && n_T~=n_df
        error('If T and df are vectors, they must have the same number of elements');
    end
end

n_samples = max(n_df,n_T);

E1 = exp(-T/T1);
if n_samples > n_T
    E1 = repmat(E1,[1,1,n_samples]);
end
E2 = exp(-T/T2);
phi = 2*pi*df.*T;	% Resonant precession, radians.

% we shoud calculate:
% E12 = [E2  O  O;...
%         O E2  O;...
%         O  O E1];
% Afp = E12*zrot(phi);
% can't do matrix multiply since E12 and Z are potentially 3D. In this case
% manual calculation is faster anyway.
O = zeros([1,1,n_samples]);
c = cos(phi);
s = sin(phi);
E2_cos = E2.*c;
E2_sin = E2.*s;

Afp = [E2_cos -E2_sin  O;...
       E2_sin  E2_cos  O;...
            O       O E1];
                                               
Bfp = [O;O;1-E1];


