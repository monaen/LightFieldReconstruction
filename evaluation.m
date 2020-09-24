%% This script evaluate the results using matlab
clear all; clc;


% parameters setting
result_folder = 'results';

% settings
lfreconslist = dir(fullfile(result_folder, 'SR', '*.mat'));
lflabellist  = dir(fullfile(result_folder, 'HR', '*.mat'));
% lflabellist = dir(fullfile(['../Data/MAT/', dataset, '/*.mat']));

meanPSNRs = [];
meanSSIMs = [];

N = length(lfreconslist);
for i = 1:N
    reconstruction = load(fullfile(result_folder, 'SR', lfreconslist(i).name));
    reconstruction = squeeze(reconstruction.data);
    reconstruction = permute(reconstruction, [3,4,1,2]);
    groundtruth = load(fullfile(result_folder, 'HR', lflabellist(i).name));
    groundtruth = squeeze(groundtruth.data);  % [height, width, sview, tview]
    groundtruth = permute(groundtruth, [3,4,1,2]);
    
    s = size(reconstruction, 1);
    t = size(reconstruction, 2);
    
    PSNR = zeros(s,t);
    SSIM = zeros(s,t);
    
    for j = 1 : s
        for k = 1 : t
            gt_img = squeeze(groundtruth(j,k,:,:));
            gt_img = im2double(gt_img);
            recons_img = squeeze(reconstruction(j,k,:,:));
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
    
end

fprintf('\nThe entire MEAN PSNR value for dataset is: %0.8f', mean(meanPSNRs));
fprintf('\nThe entire MEAN SSIM value for dataset is: %0.8f', mean(meanSSIMs));



















