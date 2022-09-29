
clear all;
close all;

gt_path='/home/weiyanyan/GAN/Datasets/val/leftImg8bit/200mm/norain';

MH_DerainNet = '../experiment/200mm_v79/results';

 
struct_model = {
          struct('model_name','MH_DerainNet','path',MH_DerainNet),...
    };


nimgs=500;nrain=1;
nmodel = length(struct_model);

psnrs = zeros(nimgs,nmodel);
ssims = psnrs;

for nnn = 1:1
    
    tp=0;ts=0;te=0;
    nstart = 0;
    for iii=nstart+1:nstart+nimgs
        for jjj=1:nrain
            fprintf(fullfile(struct_model{nnn}.path,sprintf('norain-%03d.png',iii)));
			x_true=im2double(imread(fullfile(gt_path,sprintf('norain-%03d.png',iii))));%x_true
            % x_true=im2double(imread(fullfile(gt_path,sprintf('norain-%03d_x2_HR.png',iii))));%x_true
			% x_true=im2double(imread(fullfile(gt_path,sprintf('%01d_x2_HR.png',iii))));%x_true
            x_true = rgb2ycbcr(x_true);x_true=x_true(:,:,1);
			
            %%
            x = (im2double(imread(fullfile(struct_model{nnn}.path,sprintf('norain-%03d.png',iii)))));
			%x = (im2double(imread(fullfile(struct_model{nnn}.path,sprintf('norain-%03d_x2_SR.png',iii)))));
			% x = (im2double(imread(fullfile(struct_model{nnn}.path,sprintf('%01d_x2_SR.png',iii)))));
            x = rgb2ycbcr(x);x = x(:,:,1);
            tp = mean(psnr(x,x_true));
            ts = ssim(x*255,x_true*255);
            
            psnrs(iii-nstart,nnn)=tp;
            ssims(iii-nstart,nnn)=ts;
            
            %
        end
    end
    
    fprintf('%s: psnr=%6.4f, ssim=%6.4f\n',struct_model{nnn}.model_name,mean(psnrs(:,nnn)),mean(ssims(:,nnn)));
    
end



