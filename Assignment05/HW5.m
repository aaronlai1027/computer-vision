%% 1-1 Solving Correspondence

ir = imread('/Users/aaron/Desktop/Computer Vision/Assignment05_546/BinocularStereo/tsukuba_r.ppm');
il = imread('/Users/aaron/Desktop/Computer Vision/Assignment05_546/BinocularStereo/tsukuba_l.ppm');

ir = double(rgb2gray(ir));
il = double(rgb2gray(il));
%% 1-1.1 Choose a patch size and a similarity metric

halfPatchSize = 10;
PatchSize = 2 * halfPatchSize + 1;
%pixel-wise sum of squared distances


% 1-1.2 calculate and plot the output (profile) of your similarity metric
%p1(136,83), p2(203, 304), p3(182, 119), p4(186, 160), p5(123, 224), p6(153, 338)

x = 123;
y = 224;

template = il(x-halfPatchSize:x+halfPatchSize,y-halfPatchSize:y+halfPatchSize);
[row,col] = size(il);


for j = 1:col
    minc = y - halfPatchSize;
    maxc = y + halfPatchSize;
    
    mind = 1 - minc;
    maxd = col - maxc;
    
    numPatch = maxd - mind + 1;
    Patchdiff = zeros(numPatch,1);
    
    for j = mind:maxd
        Patch = ir(x-halfPatchSize:x+halfPatchSize, minc+j:maxc+j);
        PatchIndex = j - mind +1;
        Patchdiff(PatchIndex,1) = sum(sum((template - Patch).^2));
    end
    
    plot(mind:maxd,Patchdiff),title(['P(' num2str(x) ',' num2str(y) ')']);
end

%subplot 121, imshow(uint8(il))
%subplot 122, imshow(uint8(ir))

%% 1-1.3
% In order to find the best match of correspondence for each pixel, I sort the value of sum of squared distances 
% and look for the smallest one. Besides, the disparity of pixels in two
% images are supposes to be in a reasonable range. That is to say, we should
% find the corresponding pixel whose coordinate is near the original one so
% we can set a range for disparity.

x = 153;
y = 338;

halfPatchSize = 10;
PatchSize = 2 * halfPatchSize + 1;
dispRange = 50;
template = il(x-halfPatchSize:x+halfPatchSize,y-halfPatchSize:y+halfPatchSize);
[row,col] = size(il);

for j = 1:col
    minc = y - halfPatchSize;
    maxc = y + halfPatchSize;
    
    mind = max(-dispRange,1 - minc);
    maxd = min(dispRange,col - maxc);
    
    numPatch = maxd - mind + 1;
    Patchdiff = zeros(numPatch,1);
    
    for j = mind:maxd
        Patch = ir(x-halfPatchSize:x+halfPatchSize, minc+j:maxc+j);
        PatchIndex = j - mind +1;
        Patchdiff(PatchIndex,1) = sum(sum((template - Patch).^2));
    end
    
    plot(mind:maxd,Patchdiff),title(['P(' num2str(x) ',' num2str(y) ')']);
end

%% 1-1.4 

halfPatchSize = 5;
PatchSize = 2 * halfPatchSize + 1;
dispRange = 50;
DispMap = zeros(size(il), 'single');

[row, col] = size(il);

for (i = 1 : row)
    minr = max(1, i - halfPatchSize);
    maxr = min(row, i + halfPatchSize);
    for (j = 1 : col)
		minc = max(1, j - halfPatchSize);
        maxc = min(col, j + halfPatchSize);
        
        template = il(minr:maxr, minc:maxc);
        
		mind = max(-dispRange, 1 - minc);
        maxd = min(dispRange, col - maxc);
		
		numPatch = maxd - mind + 1;
		
		PatchDiff = zeros(numPatch, 1);
		
		for (n = mind : maxd)
			Patch = ir(minr:maxr, (minc + n):(maxc + n));
			PatchInd = n - mind + 1;
			PatchDiff(PatchInd, 1) = sum(sum((template - Patch).^2));
		end
		
		[temp, sortedInd] = sort(PatchDiff);
		bestMatchInd = sortedInd(1, 1);
		disp = bestMatchInd + mind - 1;
			
        DispMap(i, j) = disp;
    end
   
	if (mod(i, 10) == 0)
		fprintf('  Image row %d / %d (%.0f%%)\n', i, row, (i / row) * 100);
	end
end
figure
imshow(DispMap.*(-1), []),axis image, colormap('jet'), colorbar;
title(['Nondynamic, Block Size =' num2str(PatchSize)]);
caxis([0 dispRange]);

%% 1-1.5
DisMapInter = interp2(DispMap,'cubic');
imshow(DisMapInter.*(-1), []),axis image, colormap('jet'), colorbar;
title(['Block Size =' num2str(PatchSize)]);
caxis([0 dispRange]);

%% 
halfPatchSize = 3;
PatchSize = 2 * halfPatchSize + 1;
dispRange = 30;
DynMap = zeros(size(il), 'single');
falseinf = 1e3; % False infinity
dispCost = falseinf*ones(size(il,2), 2*dispRange + 1, 'single');
dispPenalty = 0.5; % Penalty for disparity disagreement between pixels

for i=1:size(il,1)
    minr = max(1,i-halfPatchSize);
    maxr = min(size(il,1),i+halfPatchSize);
    dispCost(:) = falseinf;
    
    for n=1:size(il,2)
    minc = max(1,n-halfPatchSize);
    maxc = min(size(il,2),n+halfPatchSize);
    
    mind = max( -dispRange, 1-minc );
    maxd = min( dispRange, size(il,2)-maxc );
    
        for d=mind:maxd
        dispCost(n, d + dispRange + 1) = sum(sum(abs(il(minr:maxr,(minc:maxc)+d) - ir(minr:maxr,minc:maxc))));
        end
    end
    % Process scanline disparity costs with dynamic programming.
    optimalInd = zeros(size(dispCost), 'single');
    cp = dispCost(end,:);
    
    for j=size(dispCost,1)-1:-1:1
    % False infinity for this level
    cfinf = (size(dispCost,1) - j + 1)*falseinf;
    % Construct matrix for finding optimal move for each column
    % individually.
    [v,ix] = min([cfinf cfinf cp(1:end-4)+3*dispPenalty;
    cfinf cp(1:end-3)+2*dispPenalty;
    cp(1:end-2)+dispPenalty;
    cp(2:end-1);
    cp(3:end)+dispPenalty;
    cp(4:end)+2*dispPenalty cfinf;
    cp(5:end)+3*dispPenalty cfinf cfinf],[],1);
    cp = [cfinf dispCost(j,2:end-1)+v cfinf];
    % Record optimal routes.
    optimalInd(j,2:end-1) = (2:size(dispCost,2)-1) + (ix - 4);
    end
    
    % Recover optimal route.
    [~,ix] = min(cp);
    DynMap(i,1) = ix;
    
    for k=1:size(DynMap,2)-1
    DynMap(i,k+1) = optimalInd(k, max(1, min(size(optimalInd,2), round(DynMap(i,k)) ) ) );
    end
end
DynMap = DynMap - dispRange - 1;

imshow(DynMap, []),axis image, colormap('jet'), colorbar;
title(['Dynamic, Block Size =' num2str(PatchSize)]);
caxis([0 dispRange]);
%% graduate credit
I0 = imread('/Users/aaron/Desktop/Computer Vision/Assignment05_546/PhotometricStereo/psmImages/buddha/buddha.0.png');
I1 = imread('/Users/aaron/Desktop/Computer Vision/Assignment05_546/PhotometricStereo/psmImages/buddha/buddha.1.png');
I2 = imread('/Users/aaron/Desktop/Computer Vision/Assignment05_546/PhotometricStereo/psmImages/buddha/buddha.2.png');
I3 = imread('/Users/aaron/Desktop/Computer Vision/Assignment05_546/PhotometricStereo/psmImages/buddha/buddha.3.png');
I4 = imread('/Users/aaron/Desktop/Computer Vision/Assignment05_546/PhotometricStereo/psmImages/buddha/buddha.4.png');
I5 = imread('/Users/aaron/Desktop/Computer Vision/Assignment05_546/PhotometricStereo/psmImages/buddha/buddha.5.png');
I6 = imread('/Users/aaron/Desktop/Computer Vision/Assignment05_546/PhotometricStereo/psmImages/buddha/buddha.6.png');
I7 = imread('/Users/aaron/Desktop/Computer Vision/Assignment05_546/PhotometricStereo/psmImages/buddha/buddha.7.png');
I8 = imread('/Users/aaron/Desktop/Computer Vision/Assignment05_546/PhotometricStereo/psmImages/buddha/buddha.8.png');
I9 = imread('/Users/aaron/Desktop/Computer Vision/Assignment05_546/PhotometricStereo/psmImages/buddha/buddha.9.png');
I10 = imread('/Users/aaron/Desktop/Computer Vision/Assignment05_546/PhotometricStereo/psmImages/buddha/buddha.10.png');
I11 = imread('/Users/aaron/Desktop/Computer Vision/Assignment05_546/PhotometricStereo/psmImages/buddha/buddha.11.png');
Imask = imread('/Users/aaron/Desktop/Computer Vision/Assignment05_546/PhotometricStereo/psmImages/buddha/buddha.mask.png');
light = load('/Users/aaron/Desktop/Computer Vision/Assignment05_546/PhotometricStereo/Code/lighting.mat');
L = light.L;

%% red channel
I0r = double(I0(:,:,1));
I1r = double(I1(:,:,1));
I2r = double(I2(:,:,1));
I3r = double(I3(:,:,1));
I4r = double(I4(:,:,1));
I5r = double(I5(:,:,1));
I6r = double(I6(:,:,1));
I7r = double(I7(:,:,1));
I8r = double(I8(:,:,1));
I9r = double(I9(:,:,1));
I10r = double(I10(:,:,1));
I11r = double(I11(:,:,1));

%% red normal vector

for i=1:size(I0r,1)
    for j=1:size(I0r,2)
        N(i,j,1) = 0;
        N(i,j,2) = 0;
        N(i,j,3) = 0;
        if(Imask(i,j)>0)
            ixy = [I0r(i,j);I1r(i,j);I2r(i,j);I3r(i,j);I4r(i,j);I5r(i,j);I6r(i,j);...
                I7r(i,j);I8r(i,j);I9r(i,j);I10r(i,j);I11r(i,j)];
            gr = L\ixy;
            kp = norm(gr);
            Nxy = gr./kp;
            N(i,j,1) = Nxy(1);
            N(i,j,2) = Nxy(2);
            N(i,j,3) = Nxy(3);
            
            pxy(i,j) = Nxy(1)/Nxy(3);
            if(pxy(i,j) == NaN)
                pxy(i,j) = 0;
            end
            qxy(i,j) = Nxy(2)/Nxy(3);
            if(qxy(i,j) == NaN)
                qxy(i,j) = 0;
            end
        end
    end
end
quiver(1:5:size(pxy,2),1:5:size(pxy,1),pxy(1:5:end,1:5:end),qxy(1:5:end,1:5:end),10)
axis tight ij,title('surface normal vector')
%% red depth map

vdepth = zeros(size(pxy));
for j=1:size(pxy,2)
    for i=1:size(pxy,1)
        if(Imask(i,j)>0)
            vdepth(i,j) = vdepth(i-1,j) + qxy(i,j);
        end
    end
    if(Imask(i,j)>0)
        vdepth(i,j) = vdepth(i,j-1) + pxy(i,j);
    end
end
surfl(vdepth); shading interp; colormap gray; axis tight,
title('depth map')

%% red refineDepthMap

mask = Imask(:,:,1)>0;
depth = refineDepthMap(N,mask);
quiver(1:5:size(pxy,2),1:5:size(pxy,1),pxy(1:5:end,1:5:end),qxy(1:5:end,1:5:end),10)
axis tight ij
surfl(depth); shading interp; colormap gray; axis tight,
title('depth map using refineDepthMap')

%% green channel
I0g = double(I0(:,:,2));
I1g = double(I1(:,:,2));
I2g = double(I2(:,:,2));
I3g = double(I3(:,:,2));
I4g = double(I4(:,:,2));
I5g = double(I5(:,:,2));
I6g = double(I6(:,:,2));
I7g = double(I7(:,:,2));
I8g = double(I8(:,:,2));
I9g = double(I9(:,:,2));
I10g = double(I10(:,:,2));
I11g = double(I11(:,:,2));

%% green normal vector

for i=1:size(I0r,1)
    for j=1:size(I0r,2)
        if(Imask(i,j)>0)
            ixy = [I0g(i,j);I1g(i,j);I2g(i,j);I3g(i,j);I4g(i,j);I5g(i,j);I6g(i,j);...
                I7g(i,j);I8g(i,j);I9g(i,j);I10g(i,j);I11g(i,j)];
            gx = L\ixy;
            kp = norm(gx);
            Nxy = gx./kp;
            N(i,j,1) = Nxy(1);
            N(i,j,2) = Nxy(2);
            N(i,j,3) = Nxy(3);
            
            pxy(i,j) = Nxy(1)/Nxy(3);
            if(pxy(i,j) == NaN)
                pxy(i,j) = 0;
            end
            qxy(i,j) = Nxy(2)/Nxy(3);
            if(qxy(i,j) == NaN)
                qxy(i,j) = 0;
            end
        end
    end
end
quiver(1:5:size(pxy,2),1:5:size(pxy,1),pxy(1:5:end,1:5:end),qxy(1:5:end,1:5:end),10)
axis tight ij,title('surface normal vector')

%% green depth map

vdepth = zeros(size(pxy));
for j=1:size(pxy,2)
    for i=1:size(pxy,1)
        if(Imask(i,j)>0)
            vdepth(i,j) = vdepth(i-1,j) + qxy(i,j);
        end
    end
    if(Imask(i,j)>0)
        vdepth(i,j) = vdepth(i,j-1) + pxy(i,j);
    end
end
surfl(vdepth); shading interp; colormap gray; axis tight

%% green refineDepthMap 

mask = Imask(:,:,2)>0;
depth = refineDepthMap(N,mask);
quiver(1:5:size(pxy,2),1:5:size(pxy,1),pxy(1:5:end,1:5:end),qxy(1:5:end,1:5:end),10)
axis tight ij
surfl(depth); shading interp; colormap gray; axis tight

%% blue channel
I0b = double(I0(:,:,3));
I1b = double(I1(:,:,3));
I2b = double(I2(:,:,3));
I3b = double(I3(:,:,3));
I4b = double(I4(:,:,3));
I5b = double(I5(:,:,3));
I6b = double(I6(:,:,3));
I7b = double(I7(:,:,3));
I8b = double(I8(:,:,3));
I9b = double(I9(:,:,3));
I10b = double(I10(:,:,3));
I11b = double(I11(:,:,3));

%% blue normal vector

for i=1:size(I0r,1)
    for j=1:size(I0r,2)
        if(Imask(i,j)>0)
            ixy = [I0b(i,j);I1b(i,j);I2b(i,j);I3b(i,j);I4b(i,j);I5b(i,j);I6b(i,j);...
                I7b(i,j);I8b(i,j);I9b(i,j);I10b(i,j);I11b(i,j)];
            gb = L\ixy;
            kp = norm(gb);
            Nxy = gb./kp;
            N(i,j,1) = Nxy(1);
            N(i,j,2) = Nxy(2);
            N(i,j,3) = Nxy(3);
            
            pxy(i,j) = Nxy(1)/Nxy(3);
            if(pxy(i,j) == NaN)
                pxy(i,j) = 0;
            end
            qxy(i,j) = Nxy(2)/Nxy(3);
            if(qxy(i,j) == NaN)
                qxy(i,j) = 0;
            end
        end
    end
end
quiver(1:5:size(pxy,2),1:5:size(pxy,1),pxy(1:5:end,1:5:end),qxy(1:5:end,1:5:end),10)
axis tight ij

%% blue depth map

vdepth = zeros(size(pxy));
for j=1:size(pxy,2)
    for i=1:size(pxy,1)
        if(Imask(i,j)>0)
            vdepth(i,j) = vdepth(i-1,j) + qxy(i,j);
        end
    end
    if(Imask(i,j)>0)
        vdepth(i,j) = vdepth(i,j-1) + pxy(i,j);
    end
end
surfl(vdepth); shading interp; colormap gray; axis tight

%% blue refineDepthMap

mask = Imask(:,:,1)>0;
depth = refineDepthMap(N,mask);
quiver(1:5:size(pxy,2),1:5:size(pxy,1),pxy(1:5:end,1:5:end),qxy(1:5:end,1:5:end),10)
axis tight ij
surfl(depth); shading interp; colormap gray; axis tight


figure
imshow(uint8(I0r));title('red')
figure
imshow(uint8(I0g));title('green')
figure
imshow(uint8(I0b));title('blue')
