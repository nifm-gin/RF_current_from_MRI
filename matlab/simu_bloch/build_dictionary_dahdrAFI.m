%% Calculate dictionary of slice profiles for various flip angles

%% Load RF pulse information
rfvar = sinc_gauss;
gamma = 4258;                   % gyoromagnetic ratio for protons, Hz/Gauss

%% Define dictionary properties
t1  = 0.5;                      % seconds
t2s = 0.04;                     % seconds
n_pos = 201;                     % number of spatial samples (through-plane)
extent = 4;                     % extent of sampling (multiples of slice thickness)
rf_dur = 0.002;                 % RF pulse duration, seconds

%% Set AFI acquisition parameters to verify AFI signals
% Sequence a
nominal_flip1 = 45.2;           % degrees - required to calculate isodelay
nominal_flip2 = 69.3;           % degrees - required to calculate isodelay
tr1 = 0.0237;                   % seconds
n   = 2.59;
% % Sequence b
% nominal_flip1 = 21.9;           % degrees - required to calculate isodelay
% nominal_flip2 = 39.0;           % degrees - required to calculate isodelay
% tr1 = 0.0265;                   % seconds
% n   = 7.11;

te  = 0.003;
flip_angles1 = 1:1000;          % degrees - defines dictionary entries
slice_thickness = 2;            % mm

%% Flags to control output of this script
do_plot = true;

%% Code (no need to modify below this line)
r_nominal_flip = nominal_flip2 / nominal_flip1;
n_dict1 = numel(flip_angles1);  % size of dictionary

e1  = exp(-tr1/t1);
e2  = exp(-tr1*n/t1);
et2s= exp(-te/t2s);
ca1 = cos(flip_angles1*pi/180);
sa1 = sin(flip_angles1*pi/180);
ca2 = cos(r_nominal_flip * flip_angles1*pi/180);
sa2 = sin(r_nominal_flip * flip_angles1*pi/180);

satess1 = sa1 * et2s;
satess2 = sa2 * et2s;
e1e2 = e1 * e2;
sf1 = satess1 ./ (1 - e1e2 .* ca1 .* ca2);
sf2 = satess2 ./ (1 - e1e2 .* ca1 .* ca2);

afi_theory = zeros(n_dict1, n_pos, 2);
afi_theory(:, 1) = sf1 .* ((1 - e2) + (e2 - e1e2) .* ca2);
afi_theory(:, 2) = sf2 .* ((1 - e1) + (e1 - e1e2) .* ca1);

%% Calculate RF and gradient envelopes with correct scale
% We will have numel(am_shape) RF and gradient samples for the RF pulse
% with slice select gradient, followed by a single sample where rf=0 with
% the refocusing gradient (we are neglecting relaxation).
am_shape = rfvar.shape;
am_shape = am_shape / (rfvar.amp_int * rf_dur); % Normalize to integral=1
% scale envelope to produce a nominal 1° flip angle
rf_per_degree = am_shape * (1/360) / gamma;  % Gauss
rf_per_degree = [rf_per_degree; 0]; % add null sample at the end for refoc

dt = rf_dur / rfvar.n_samples; % duration of each RF/grad sample (seconds)

% calculate amplitude of slice select gradient
g_s = rfvar.tbp / (rf_dur*gamma*slice_thickness); % Gauss/mm
g_s = 10 * g_s;                 % Gauss/cm
% calculate reference position of the pulse (from the start) for refocusing
ref1 = rf_dur * rfvar.sym; % seconds
refoc_dur1 = rf_dur - ref1;       % time after the reference position
g_refoc1 = -g_s * refoc_dur1 / dt;% g_refoc*dt = -g_s*refoc_dur (seconds)
grad1 = [g_s*ones(rfvar.n_samples,1); g_refoc1];
ref2 = rf_dur * rfvar.sym; % seconds
refoc_dur2 = rf_dur - ref2;       % time after the reference position
g_refoc2 = -g_s * refoc_dur2 / dt;% g_refoc*dt = -g_s*refoc_dur (seconds)
grad2 = [g_s*ones(rfvar.n_samples,1); g_refoc2];

% calculate positions to space them symetrically. Center of slice: pos=0
pos = slice_thickness * extent * ((1:n_pos)-(n_pos+1)/2)/n_pos;  % mm

% initialize dictionary and also a matrix with magnetizations for debugging
dict_fa1 = zeros(n_dict1, n_pos);
dict_ph1 = zeros(n_dict1, n_pos);
dict_fa2 = zeros(n_dict1, n_pos);
dict_ph2 = zeros(n_dict1, n_pos);
Mxyz1 = zeros(n_dict1, n_pos, 3);
Mxyz2 = zeros(n_dict1, n_pos, 3);
afi_signals = zeros(n_dict1, n_pos, 2);

t_start = tic;
for i_dict1 = 1:n_dict1
    this_flip1 = flip_angles1(i_dict1);% degrees
    rf1 = rf_per_degree * this_flip1; % Scale RF pulse in amplitude
    
    % Apply RF pulse along the y-axis (imaginary B1). In a right-handed 
    % coordinate system this produces real-valued transverse magnetization.
    [Mx1,My1,Mz1] = sliceprofile(1i*rf1,grad1,dt,t1,t2s,pos);
    Mxyz1(i_dict1,:,:) = cat(3, Mx1', My1', Mz1');
    
    % calculate actual flip angles from the resulting magnetization
    fa1 = atan2(sqrt(Mx1.^2+My1.^2), Mz1)'; % rad
    % calculate actual RF phases from the resulting magnetization
    ph1 = atan2(My1, Mx1)';        % rad
    
    dict_fa1(i_dict1,:) = fa1 * 180/pi;% degrees
    dict_ph1(i_dict1,:) = ph1 * 180/pi;% degrees
    
    this_flip2 = r_nominal_flip * this_flip1;% degrees
    rf2 = rf_per_degree * this_flip2; % Scale RF pulse in amplitude
    
    % Apply RF pulse along the y-axis (imaginary B1). In a right-handed 
    % coordinate system this produces real-valued transverse magnetization.
    [Mx2,My2,Mz2] = sliceprofile(1i*rf2,grad2,dt,t1,t2s,pos);
    Mxyz2(i_dict1,:,:) = cat(3, Mx2', My2', Mz2');
    
    % calculate actual flip angles from the resulting magnetization
    fa2 = atan2(sqrt(Mx2.^2+My2.^2), Mz2)'; % rad
    % calculate actual RF phases from the resulting magnetization
    ph2 = atan2(My2, Mx2)';        % rad
        
    dict_fa2(i_dict1,:) = fa2 * 180/pi;% degrees
    dict_ph2(i_dict1,:) = ph2 * 180/pi;% degrees
    
    % calculate complex afi signals across the slice
    afi_signals(i_dict1, : ,1) = et2s*exp(1i*ph1).*sin(fa1).*(1-e2+(1-e1)*e2*cos(fa2))./(1-e1*e2*cos(fa1).*cos(fa2)); 
    afi_signals(i_dict1, : ,2) = et2s*exp(1i*ph2).*sin(fa2).*(1-e1+(1-e2)*e1*cos(fa1))./(1-e1*e2*cos(fa1).*cos(fa2)); 
end
t_end = toc(t_start);
fprintf('Duration of simulation: %f s\n', t_end);

% average AFI signals across simulation domain
% scale with extent, since only 1/extent is inside the slice
afi_sim = squeeze(mean(afi_signals,2)) * extent;

if do_plot
    % average magnetization across simulation domain
    % scale with extent, since only 1/extent is inside the slice
    mean_Mxyz1 = squeeze(mean(Mxyz1,2)) * extent;
    figure; 
    subplot(2,1,1);
    plot(flip_angles1, mean_Mxyz1(:,1), 'b'); 
    hold on; 
    plot(flip_angles1, sin(flip_angles1*pi/180),'r'); 
    ylabel('integral(M_x)')
    xlabel('Flip angle [deg]')
    legend('Bloch w/ slice profile','sin(alpha)');
    title('Average magnetization across slice')
    grid on
    subplot(2,1,2);
    plot(flip_angles1, squeeze(mean(Mxyz1(:,:,3),2)), 'b'); 
    hold on; 
    plot(flip_angles1, 1+(cos(flip_angles1*pi/180)-1)/extent,'r'); 
    ylabel('integral(M_z)')
    xlabel('Flip angle [deg]')
    legend('Bloch w/ slice profile','1 + (cos(alpha)-1)/extent');
    grid on
    
    figure; 
    subplot(2,1,1);
    plot(flip_angles1, real(afi_sim(:,1)), 'b'); 
    hold on; 
    plot(flip_angles1, afi_theory(:,1),'r'); 
    plot(flip_angles1, imag(afi_sim(:,1)), 'g');  % should be zero, plot to check
    ylabel('AFI S_1')
    xlabel('Flip angle [deg]')
    legend('Bloch w/ slice profile','Analytical');
    title('AFI signals')
    grid on
    subplot(2,1,2);
    plot(flip_angles1, real(afi_sim(:,2)), 'b'); 
    hold on; 
    plot(flip_angles1, afi_theory(:,2),'r'); 
    plot(flip_angles1, imag(afi_sim(:,2)), 'g');  % should be zero, plot to check
    ylabel('AFI S_2')
    xlabel('Flip angle [deg]')
    grid on
    
    plot_flip_idx = 90;
    figure;
    subplot(2,1,1);
    plot(pos,Mxyz1(plot_flip_idx,:,1),'b');
    hold on;
    plot(pos,Mxyz1(plot_flip_idx,:,2),'r');
    plot(pos,Mxyz1(plot_flip_idx,:,3),'g');
    grid on
    legend('M_x','M_y','M_z')
    xlabel('Position [mm]')
    ylabel('Magnetization')
    title(sprintf('Flip angle: %d°',flip_angles1(plot_flip_idx)))
    subplot(2,1,2);
    plot(pos, dict_fa1(plot_flip_idx,:),'b');
    hold on
    plot(pos, dict_ph1(plot_flip_idx,:),'r');
    legend('Flip angle','B_1 phase')
    xlabel('Position [mm]')
    ylabel('Flip angle / phase [deg]')
    grid on
end

real_afi_sim = real(afi_sim);
fname_base = sprintf('dahdrafi_sim_%.2f_%.2f_%.2f_%.2f_%.0f',nominal_flip1,nominal_flip2,tr1*1e3,n,t1*1e3);
save(fname_base + "_real", 'real_afi_sim', 'flip_angles1');
imag_afi_sim = imag(afi_sim);
save(fname_base + "_imag", 'imag_afi_sim', 'flip_angles1');


