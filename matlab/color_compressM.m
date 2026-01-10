function color_compressM(image_path)

% - If no image_path: scans ../pictures and lets you pick
% - Asks percent of coefficients to keep
% - Shows a 2x2 grid: Original, FFT, DCT, DWT (if available)

  
    % Pick an image if none given
  
    if nargin < 1 || isempty(image_path)
        scriptDir = fileparts(mfilename('fullpath'));
        picturesDir = fullfile(scriptDir, '..', 'pictures');

        if exist(picturesDir, 'dir') ~= 7
            error('Pictures folder not found at: %s', picturesDir);
        end

        exts = {'*.png','*.jpg','*.jpeg','*.webp','*.bmp','*.tif','*.tiff'};
        files = [];
        for e = exts
            files = [files; dir(fullfile(picturesDir, e{1}))]; %#ok<AGROW>
        end

        if isempty(files)
            error('No images found in: %s', picturesDir);
        end

        fprintf('Available images in pictures/:\n');
        for k = 1:numel(files)
            fprintf('  %d: %s\n', k, files(k).name);
        end

        choice = input('Enter index or filename to use (press Enter to pick 1): ','s');
        if isempty(choice), choice = '1'; end

        if all(isstrprop(choice, 'digit'))
            idx = str2double(choice);
            if idx < 1 || idx > numel(files)
                error('Index out of range');
            end
            image_path = fullfile(picturesDir, files(idx).name);
        else
            candidate = fullfile(picturesDir, choice);
            if exist(candidate, 'file') == 2
                image_path = candidate;
            else
                found = false;
                for k = 1:numel(files)
                    [~, nm, ~] = fileparts(files(k).name);
                    if strcmpi(files(k).name, choice) || strcmpi(nm, choice)
                        image_path = fullfile(picturesDir, files(k).name);
                        found = true;
                        break;
                    end
                end
                if ~found
                    error('Image not found: %s', choice);
                end
            end
        end
    end

   
    % Read image (basic)
 
    if exist(image_path, 'file') ~= 2
        error('File not found: %s', image_path);
    end
    img = imread(image_path);
    if size(img,3) == 3
        img = rgb2gray(img);
    end
    img = double(img);

    % Ask for percentage 
    
    fprintf('Enter percentage of coefficients to keep.\n');
    pct = input('Enter X (1–100), or press Enter for default (5%): ','s');

    if isempty(pct)
        keep_ratio = 0.05;
    else
        pct = str2double(pct);
        if isnan(pct) || pct <= 0 || pct > 100
            error('Percentage must be a number between 1 and 100.');
        end
        keep_ratio = pct / 100;
    end
    fprintf('Using %.1f%% coefficients.\n\n', 100*keep_ratio);

    % -------------------------
    % FFT compression (fast percentile cutoff)
    % -------------------------
    F = fftshift(fft2(img));
    magF = abs(F);
    cutoffF = prctile(magF(:), 100*(1 - keep_ratio));
    F(magF < cutoffF) = 0;
    img_fft = real(ifft2(ifftshift(F)));

    % -------------------------
    % DCT compression (fast percentile cutoff)
    % -------------------------
    % dct2/idct2 require Image Processing Toolbox OR can exist in base depending on version
    try
        C = dct2(img);
        magC = abs(C);
        cutoffC = prctile(magC(:), 100*(1 - keep_ratio));
        C(magC < cutoffC) = 0;
        img_dct = idct2(C);
        dct_ok = true;
    catch
        img_dct = zeros(size(img));
        dct_ok = false;
    end

    % -------------------------
    % DWT compression (Wavelet Toolbox)
    % -------------------------
    dwt_ok = false;
    img_dwt = zeros(size(img));
    dwt_msg = '';

    if exist('wavedec2','file') == 2 && exist('waverec2','file') == 2
        try
            wname = 'db1'; % Haar-like; change to 'db2', 'sym4', etc. if you want
            level = wmaxlev(size(img), wname);  % max safe level
            level = max(1, min(level, 4));      % cap to keep it fast/robust

            [c,s] = wavedec2(img, level, wname); % c is 1-D coefficient vector
            magc = abs(c);

            cutoffW = prctile(magc(:), 100*(1 - keep_ratio));
            c_comp = c;
            c_comp(magc < cutoffW) = 0;

            img_dwt = waverec2(c_comp, s, wname);
            img_dwt = img_dwt(1:size(img,1), 1:size(img,2)); % just in case
            dwt_ok = true;
        catch ME
            dwt_ok = false;
            dwt_msg = ME.message;
        end
    end

    % -------------------------
    % Display (clip safely)
    % -------------------------
    img_display     = uint8(min(max(img, 0), 255));
    fft_display     = uint8(min(max(img_fft, 0), 255));
    dct_display     = uint8(min(max(img_dct, 0), 255));
    dwt_display     = uint8(min(max(img_dwt, 0), 255));

    figure('Name','Transform Compression Comparison (2x2)','NumberTitle','off');

    subplot(2,2,1);
    imshow(img_display);
    title('Original');

    subplot(2,2,2);
    imshow(fft_display);
    title(sprintf('FFT (kept %.1f%%)', 100*keep_ratio));

    subplot(2,2,3);
    if dct_ok
        imshow(dct_display);
        title(sprintf('DCT (kept %.1f%%)', 100*keep_ratio));
    else
        axis off;
        text(0.5,0.5,'DCT not available','HorizontalAlignment','center');
        title('DCT');
    end

    subplot(2,2,4);
    if dwt_ok
        imshow(dwt_display);
        title(sprintf('DWT (kept %.1f%%)', 100*keep_ratio));
    else
        axis off;
        if isempty(dwt_msg)
            text(0.5,0.5,'DWT not available (Wavelet Toolbox needed)','HorizontalAlignment','center');
        else
            text(0.5,0.5,{'DWT failed:', dwt_msg},'HorizontalAlignment','center');
        end
        title('DWT');
    end
end
