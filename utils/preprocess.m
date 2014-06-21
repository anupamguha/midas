Path1='images\';
Files = dir(fullfile(Path1,'*.png'));
hold all;
for k=1:length(Files)
    disp(Files(k).name);
    edge_image=imread(fullfile(Path1,Files(k).name));
    bw_image = im2bw(edge_image,.01);
    bw_image = bwmorph(bw_image, 'dilate');
    bw_image = bwmorph(bw_image,'thin',inf);
    bw_image = bw_image-bwmorph(bw_image,'branchpoints',Inf);
   
    
    [p q] = size(bw_image);
    
    b = bw_image;
    for i=1:p
        for j=1:q
            if (i<3 || i>p-3 || j<3 || j>q-3)
                b(i,j)=0;
            end
        end
    end
    for i=2:p-1
        for j=2:q-1
            if b(i-1,j-1)+b(i-1,j)+b(i-1,j+1)+b(i,j-1)+b(i,j)+b(i,j+1)+b(i+1,j-1)+b(i+1,j)+b(i+1,j+1)>4
                b(i-1,j-1)=0;
                b(i-1,j)=0;b(i-1,j+1)=0;b(i,j-1)=0;b(i,j)=0;b(i,j+1)=0;b(i+1,j-1)=0;
                b(i+1,j)=0;b(i+1,j+1)=0;
            end
        end
    end
    
    
    
    Inew = zeros(p,q);
    [L, num] = bwlabel(b, 8);
    disp(num);
    pixelval = zeros(1,num);
    for ind = 1:num
        l = 0;
        pixelcount = 0;
        for i=1:p
            for j = 1:q
                if L(i,j)==ind
                    pixelcount = pixelcount + double(edge_image(i,j));
                    l = l + 1;
                end
            end
        end
        pixelval(ind) = pixelcount/l;
        if(pixelval(ind)>2.5*mean(pixelval))
            for i=1:p
                for j = 1:q
                    if L(i,j)==ind
                        Inew(i,j) = 1;                   
                    end
                end
            end
        end
        
        
    end
    Inew = bwareaopen(Inew, 200);
    [x, y] = bwlabel(Inew, 8);
    disp(y);

    s = Files(k).name;
    imwrite(Inew,['processed\' s]);

    
end
