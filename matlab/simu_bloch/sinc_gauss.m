function rfvar = sinc_gauss
% rfvar = sinc_gauss
%
% This function implements an asymetric sinc_gauss RF pulse.
%
% Output:
% rfvar - structure with all the relevant pulse information, including
%         shape.

% 2025 - Jan Warnking

n_intervals = 100;
n_samples = n_intervals + 1;
n_null_left = 2;
n_null_right = 1;
n_null_total = n_null_left + n_null_right;
f_center = n_null_left / n_null_total;
i_center = ceil(n_intervals * f_center);
index = (0:n_intervals)' - i_center;

t0 = n_intervals / n_null_total;        % Time between two nulls of the sinc
fwhm_sinc = 1 / t0;                     % FWHM of the sinc in the freq. domain
x = pi * fwhm_sinc * index;             % Scaled time argument
am_samples_sinc = sin(x) ./ x;          % Calculate sinc pulse
am_samples_sinc(index==0) = 1;

% Choose FWHM of the Gaussian apodization function in the frequency domain.
% The freq. response of the sinc will be convolved with a Gaussian of this
% FWHM. Choose a fairly low FWHM to not deteriorate the response of the
% sinc too much.
fwhm_gauss = fwhm_sinc / 4;
x = (pi * fwhm_gauss / (2 * sqrt(log(2)))) * index; % Scaled time argument
am_samples_gauss = exp(-x.^2);          % Calculate Gauss apodization

% Calculate the sinc-Gauss pulse
am_samples_sg = am_samples_sinc .* am_samples_gauss;

rfvar = struct(...
    'amp_int', mean(am_samples_sg), ... % amplitude integral: int(b1)
    'sym', f_center, ...                % fractional position of rf center (symmetric: 0.5)
    'tbp', n_samples * fwhm_sinc, ...   % TBP
    'n_samples', n_samples, ...
    'shape', am_samples_sg);
