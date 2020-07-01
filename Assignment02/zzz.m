p = imread('/Users/aaron/Desktop/0423.jpg');

p = double(rgb2gray(p));
p1 = p(190:420,300:490);
p2 = p(410:600,730:870);


pr1 = imresize(p1, 0.8, 'bicubic');
pr2 = imresize(p2, 1.2, 'bicubic');

i1 = flipdim(pr1 ,2);   
i2 = flipdim(pr2 ,2);   

pp = zeros(size(p));
pp(1+190:size(i2,1)+190, 1+310:size(i2,2)+310)= i2;
pp(1+415:size(i1,1)+415, 1+725:size(i1,2)+725)= i1;


L = 14;
gaussf = fspecial('gaussian',L,5);
fAP = zeros(L,L);
fAP(round(15/2),round(15/2)) = 1;
fHP = fAP - gaussf;

imAP = imfilter(pp,fAP,'symmetric');
imLP = imfilter(p,gaussf,'symmetric');
imHP = imfilter(pp,fHP,'symmetric');

im = imLP + imHP;
imshow(uint8(im))