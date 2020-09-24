%% This script evaluate the results using matlab
clear all; clc;


% parameters setting
dataset = 'HCI_old';
style = '2X(HDRN)';
savefolder = 'Matlab_Results';

% settings
lfreconslist = dir(fullfile('Results', style, dataset, 'SR', '*.mat'));
lflabellist  = dir(fullfile('Results', style, dataset, 'HR', '*.mat'));
% lflabellist = dir(fullfile(['../Data/MAT/', dataset, '/*.mat']));

meanPSNRs = [];
meanSSIMs = [];

if exist(fullfile(savefolder,style,dataset,[style '.txt']))
    fileID = fopen(fullfile(savefolder,style,dataset,[style '.txt']), 'a+');
    % fprintf(fileID, 'Reconstruction results on: %s with scale %d\n\n', dataset, mf);
else
    mkdir(fullfile(savefolder,style,dataset));
    fileID = fopen(fullfile(savefolder,style,dataset,[style '.txt']), 'w');
    fprintf(fileID, '[HDRN] Reconstruction results on: %s with scale %s\n\n', dataset, style);
end

N = length(lfreconslist);
for i = 1:N
    reconstruction = load(fullfile('Results', style, dataset, 'SR', lfreconslist(i).name));
    reconstruction = squeeze(reconstruction.data);
    groundtruth = load(fullfile('Results', style, dataset, 'HR', lflabellist(i).name));
    groundtruth = squeeze(groundtruth.data);  % [height, width, sview, tview]
    groundtruth = permute(groundtruth, [3,4,1,2]);
    
    s = size(reconstruction, 3);
    t = size(reconstruction, 4);
    
    PSNR = zeros(s,t);
    SSIM = zeros(s,t);
    
    for j = 1 : s
        for k = 1 : t
            % fprintf('s:%g t:%g\n', j,k);
            gt_img = squeeze(groundtruth(j,k,:,:));
            gt_img = im2double(gt_img);
            % gt_img = gt_img(4:end-3,4:end-3);
            recons_img = squeeze(reconstruction(:,:,j,k));
            recons_img = im2double(recons_img);
            tem_psnr = psnr(gt_img, recons_img);
            tem_ssim = ssim(gt_img, recons_img);
            PSNR(j,k) = tem_psnr;
            SSIM(j,k) = tem_ssim;
        end
    end
    
    meanPSNRs(end+1) = mean(PSNR(:));
    meanSSIMs(end+1) = mean(SSIM(:));
    
    
    %%% print to command window
    fprintf('[%s]\n', lfreconslist(i).name(1:end-4));
    fprintf('[PSNR]  LFHDRN:  %0.8f\t', mean(PSNR(:)));
    fprintf('[SSIM]  LFHDRN:  %0.8f\n', mean(SSIM(:)));
    fprintf('-----------------------------------------------\n');
    
    %%% print file
    fprintf(fileID, '[%s]\n', lfreconslist(i).name(1:end-4));
    fprintf(fileID, '[PSNR]  LFHDRN:  %0.8f\t', mean(PSNR(:)));
    fprintf(fileID, '[SSIM]  LFHDRN:  %0.8f\n', mean(SSIM(:)));
    fprintf(fileID, '-----------------------------------------------\n');
end


fprintf(fileID, '\nThe entire MEAN PSNR value for dataset [%s] is: %0.8f', dataset, mean(meanPSNRs) );
fprintf(fileID, '\nThe entire MEAN SSIM value for dataset [%s] is: %0.8f', dataset, mean(meanSSIMs) );
fclose(fileID);



















