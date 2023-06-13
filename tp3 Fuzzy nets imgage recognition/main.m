%https://www.mathworks.com/videos/getting-started-with-fuzzy-logic-toolbox-part-3-68766.html
clear
clc
% IMAGE LOAD AND PREPROCESS : 
% - remove black excess
% - resize all img to 192x192
all_class = {};
for i = 1:6
    % get all imgs in class folder i
    jpgFilename = ['Class ' num2str(i) '/*.jpg'];
    disp(jpgFilename)
    imagefiles = dir(jpgFilename);
    nfiles = length(imagefiles);

    all_img_of_class = {};
    for j = 1:nfiles
        % load image
        imageData = imread(['Class ' num2str(i) '/' imagefiles(j).name]);
        % find where colour
        nonBlackMask = imageData(:, :, 1) >= 50 | imageData(:, :, 2) >= 50 | imageData(:, :, 3) >= 50;
        % take largest region only.
        nonBlackMask = bwareafilt(nonBlackMask, 1);
        % measure the Bounding Box
        props = regionprops(nonBlackMask, 'BoundingBox');
        % crop
        imageData = imcrop(imageData, props.BoundingBox);
        % resize
        imageData = imresize(imageData,[192 192]);

        blackmask = repmat(all(~imageData,3),[1 1 1]); %mask black parts
        imageData(blackmask) = nan; %turn them nan

        imshow(imageData)
        % separate RGB 
        r_channel = imageData(:,:,1);
        g_channel = imageData(:,:,2);
        b_channel = imageData(:,:,3);
        % convert to gray scale
        gray_img = rgb2gray(imageData);
        % get indexes where black
        idx = gray_img == 0;
        % calculate average RGB of the region that isnt black
        avg_R = uint8(mean(r_channel(~idx),'omitnan') );
        avg_G = uint8(mean(g_channel(~idx),'omitnan') );
        avg_B = uint8(mean(b_channel(~idx),'omitnan') );

        % save in a sub array
        all_img_of_class{j} = [avg_R avg_G avg_B];

    end
    % save img array within class array
    all_class{i} = all_img_of_class';
end
save('dataset_cell_array','all_class')

imgs_train = {};
imgs_test = {};
for i = 1:6
    s = length(all_class{i});
    s = round(s*0.7);

    imgs_train{i} = all_class{i}(1:s,:); %#ok<*SAGROW> 
    imgs_test{i} = all_class{i}(s+1:end,:);
end 

for i = 1:6
    fprintf('CLASS %i\n', i)
    disp('AVG')
    r = mean(cellfun(@(v)v(1), imgs_train{i}));
    g = mean(cellfun(@(v)v(2), imgs_train{i}));
    b = mean(cellfun(@(v)v(3), imgs_train{i}));
    disp([r g b])
    disp('MAX')
    r = max(cellfun(@(v)v(1), imgs_train{i}));
    g = max(cellfun(@(v)v(2), imgs_train{i}));
    b = max(cellfun(@(v)v(3), imgs_train{i}));
    disp([r g b])
    disp('MIN')
    r = min(cellfun(@(v)v(1), imgs_train{i}));
    g = min(cellfun(@(v)v(2), imgs_train{i}));
    b = min(cellfun(@(v)v(3), imgs_train{i}));
    disp([r g b])
    disp('STD DEV')
    r = std2(cellfun(@(v)v(1), imgs_train{i}));
    g = std2(cellfun(@(v)v(2), imgs_train{i}));
    b = std2(cellfun(@(v)v(3), imgs_train{i}));
    disp([r g b])

    %disp('AVG TEST')
    %r = mean(cellfun(@(v)v(1), imgs_test{i}));
    %g = mean(cellfun(@(v)v(2), imgs_test{i}));
    %b = mean(cellfun(@(v)v(3), imgs_test{i}));
    %disp([r g b])
end 

fis_mam9 = readfis('9rule_mam.fis');
fis_sug9 = readfis('9rule_sug.fis');
fis_mam25 = readfis('25rule_mam.fis');
fis_sug25 = readfis('25rule_sug.fis');

out = {};
a = [];
fprintf('\n 9rule_mam')
for i = 1:6
    out{i} = round(evalfis( fis_mam9, double(cell2mat(imgs_test{i})) ));
    s = size(out{i});
    diff = find(out{i} ~= i);
    total_correct = 100 -( size(diff) * 100 ) / s;
    fprintf('\nCLASS %i correct predictions: ', i)
    fprintf('%d %%',  round(total_correct))
    a = [a round(total_correct)];
end 
fprintf('\nAverage correct predictions: %d %%',  round(mean(a)))
fprintf('\n')

out = {};
a = [];
fprintf('\n 9rule_sug')
for i = 1:6
    out{i} = round(evalfis( fis_sug9, double(cell2mat(imgs_test{i})) ));
    s = size(out{i});
    diff = find(out{i} ~= i);
    total_correct = 100 -( size(diff) * 100 ) / s;
    fprintf('\nCLASS %i correct predictions: ', i)
    fprintf('%d %%', round(total_correct))
    a = [a round(total_correct)]; %#ok<*AGROW> 
end 
fprintf('\nAverage correct predictions: %d %%',  round(mean(a)))
fprintf('\n')


out = {};
a = [];
fprintf('\n 25rule_mam')
for i = 1:6
    out{i} = round(evalfis( fis_mam25, double(cell2mat(imgs_test{i})) ));
    s = size(out{i});
    diff = find(out{i} ~= i);
    total_correct = 100 -( size(diff) * 100 ) / s;
    fprintf('\nCLASS %i correct predictions: ', i)
    fprintf('%d %%', round(total_correct))
    a = [a round(total_correct)]; %#ok<*AGROW> 
end 
fprintf('\nAverage correct predictions: %d %%',  round(mean(a)))
fprintf('\n')

out = {};
a = [];
fprintf('\n 25rule_sug')
for i = 1:6
    out{i} = round(evalfis( fis_sug25, double(cell2mat(imgs_test{i})) ));
    s = size(out{i});
    diff = find(out{i} ~= i);
    total_correct = 100 -( size(diff) * 100 ) / s;
    fprintf('\nCLASS %i correct predictions: ', i)
    fprintf('%d %%', round(total_correct))
    a = [a round(total_correct)]; %#ok<*AGROW> 
end 
fprintf('\nAverage correct predictions: %d %%',  round(mean(a)))
fprintf('\n')
















