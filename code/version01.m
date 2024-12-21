%% Clear workspace and set up environment
clear all; close all; clc
%%



%% Define input files
files = {'aspect.tif', 'curvature.tif', 'distanceFromRoads.tif', ...
         'distanceFromStreams.tif', 'elevation.tif', 'lulc.tif', ...
         'ndvi.tif', 'planCurvature.tif', 'precipitation.tif', ...
         'profileCurvature.tif', 'relativeRelief.tif', 'slope.tif'};

%% Read reference landslide inventory
fprintf('Reading landslide inventory...\n');
[target, R] = readgeoraster('/MATLAB Drive/Udemy/features/landslideInventory.tif');
info = readgeoraster('/MATLAB Drive/Udemy/features/landslideInventory.tif');
[ref_rows, ref_cols] = size(target);
target = double(target);

fprintf('Reference dimensions from landslide inventory: [%d, %d]\n', ref_rows, ref_cols);

%% Initialize data matrix
num_pixels = ref_rows * ref_cols;
num_features = length(files);
X = zeros(num_pixels, num_features);

%% Process input files
fprintf('\nProcessing input files...\n');
for i = 1:length(files)
    try
        % Read current file
        [current_data, ~] = geotiffread(files{i});
        current_info = geotiffinfo(files{i});
        current_data = double(current_data);
        
        % Get current dimensions
        [curr_rows, curr_cols] = size(current_data);
        fprintf('Processing %s - Original dimensions: [%d, %d]\n', ...
                files{i}, curr_rows, curr_cols);
        
        % Resample if dimensions don't match
        if curr_rows ~= ref_rows || curr_cols ~= ref_cols
            fprintf('Resampling %s to match reference dimensions...\n', files{i});
            
            % Resample using imresize
            current_data = imresize(current_data, [ref_rows, ref_cols], 'bicubic');
            
            % Save resampled file with original CRS info
            [~, name, ext] = fileparts(files{i});
            resampled_filename = [name '_resampled' ext];
            
            % Use the original file's GeoKeyDirectoryTag if available
            if isfield(current_info, 'GeoTIFFTags') && isfield(current_info.GeoTIFFTags, 'GeoKeyDirectoryTag')
                geotiffwrite(resampled_filename, current_data, R, ...
                    'GeoKeyDirectoryTag', current_info.GeoTIFFTags.GeoKeyDirectoryTag);
            else
                % Try using CoordRefSysCode if available
                if isfield(info, 'CoordRefSysCode')
                    geotiffwrite(resampled_filename, current_data, R, ...
                        'CoordRefSysCode', info.CoordRefSysCode);
                else
                    % If no CRS info available, just store the data
                    imwrite(current_data, resampled_filename);
                    warning('No coordinate reference system information available for %s', files{i});
                end
            end
            
            fprintf('Saved resampled file as: %s\n', resampled_filename);
        end
        
        % Normalize data to [0,1] range
        min_val = min(current_data(:));
        max_val = max(current_data(:));
        if max_val > min_val
            current_data = (current_data - min_val) / (max_val - min_val);
        else
            current_data = zeros(size(current_data));
        end
        
        % Store in pre-allocated matrix
        X(:, i) = reshape(current_data, [], 1);
        
        fprintf('Successfully processed %s\n', files{i});
        
    catch ME
        fprintf('Error processing file %s: %s\n', files{i}, ME.message);
        rethrow(ME);
    end
end

%% Prepare target data
Y = reshape(target, [], 1);

%% Remove NaN values
valid_idx = ~any(isnan(X), 2) & ~isnan(Y);
X = X(valid_idx, :);
Y = Y(valid_idx);

%% Display dataset statistics
fprintf('\nDataset statistics:\n');
fprintf('Number of samples: %d\n', size(X, 1));
fprintf('Number of features: %d\n', size(X, 2));
fprintf('Proportion of landslide pixels: %.2f%%\n', 100 * sum(Y) / length(Y));

%% Split data into training, validation, and testing sets
rng(1); % For reproducibility
num_samples = size(X, 1);
idx = randperm(num_samples);

train_size = round(0.7 * num_samples);
val_size = round(0.15 * num_samples);

train_idx = idx(1:train_size);
val_idx = idx(train_size+1:train_size+val_size);
test_idx = idx(train_size+val_size+1:end);

X_train = X(train_idx, :);
Y_train = Y(train_idx);
X_val = X(val_idx, :);
Y_val = Y(val_idx);
X_test = X(test_idx, :);
Y_test = Y(test_idx);

%% Define and configure neural network
fprintf('\nConfiguring neural network...\n');
hidden_layers = [64, 32, 16]; % Cascading architecture
net = fitnet(hidden_layers);

% Basic configuration
net.divideFcn = 'dividetrain';  % We'll handle the division manually
net.trainFcn = 'trainscg';
net.performFcn = 'crossentropy';

% Set transfer functions
for i = 1:length(hidden_layers)
    net.layers{i}.transferFcn = 'tansig';
end
net.layers{end}.transferFcn = 'logsig';

% Training parameters
net.trainParam.epochs = 100;
net.trainParam.min_grad = 1e-6;
net.trainParam.max_fail = 20;
net.trainParam.showWindow = true;

%% Train neural network
fprintf('\nTraining neural network...\n');

% Create validation and test data structures
valData.X = X_val';
valData.Y = Y_val';
valData.T = Y_val';  % Target data

testData.X = X_test';
testData.Y = Y_test';
testData.T = Y_test';  % Target data

% Configure network to use our custom validation and test sets
net.divideFcn = 'dividetrain';  % Use all data for training
net.divideMode = 'sample';      % Divide up samples (not timesteps)
net.divideParam.trainRatio = 1; % Use all data for training
net.divideParam.valRatio = 0;   % No automatic validation split
net.divideParam.testRatio = 0;  % No automatic test split

% Train the network with manual validation
[net, tr] = train(net, X_train', Y_train', [], valData);

% You can enable parallel processing and GPU if available using view
if paralleltoolbox_available
    parallelView = parallel.pool.DataQueue;
    net.useParallel = 'yes';
end

if gpuDeviceCount > 0
    net.useGPU = 'yes';
end

%% Train neural network
fprintf('\nTraining neural network...\n');
[net, tr] = train(net, X_train', Y_train', ...
    'useParallel', 'yes', ...
    'useGPU', 'yes', ...
    'ValidationData', {X_val', Y_val'}, ...
    'TestData', {X_test', Y_test'});

%% Evaluate model performance
fprintf('\nEvaluating model performance...\n');
Y_pred = net(X_test')';
Y_pred = round(Y_pred);

% Calculate metrics
confusion_mat = confusionmat(Y_test, Y_pred);
accuracy = sum(diag(confusion_mat)) / sum(confusion_mat(:));
precision = confusion_mat(2,2) / sum(confusion_mat(:,2));
recall = confusion_mat(2,2) / sum(confusion_mat(2,:));
f1_score = 2 * (precision * recall) / (precision + recall);

% Display results
fprintf('\nTest Set Performance Metrics:\n');
fprintf('Accuracy: %.4f\n', accuracy);
fprintf('Precision: %.4f\n', precision);
fprintf('Recall: %.4f\n', recall);
fprintf('F1 Score: %.4f\n', f1_score);

%% Generate and save susceptibility map
fprintf('\nGenerating susceptibility map...\n');
susceptibility_map = zeros(ref_rows, ref_cols);
valid_pixels = ~any(isnan(X), 2);
predictions = net(X(valid_pixels, :)')';
temp_map = reshape(zeros(size(X, 1), 1), [], 1);
temp_map(valid_pixels) = predictions;
susceptibility_map = reshape(temp_map, ref_rows, ref_cols);

% Save map with CRS information
if isfield(info, 'GeoTIFFTags') && isfield(info.GeoTIFFTags, 'GeoKeyDirectoryTag')
    geotiffwrite('landslide_susceptibility_map.tif', susceptibility_map, R, ...
        'GeoKeyDirectoryTag', info.GeoTIFFTags.GeoKeyDirectoryTag);
else
    geotiffwrite('landslide_susceptibility_map.tif', susceptibility_map, R);
end

%% Plot ROC curve
fprintf('\nGenerating ROC curve...\n');
[X_roc,Y_roc,~,AUC] = perfcurve(Y_test, net(X_test')', 1);
figure;
plot(X_roc,Y_roc);
xlabel('False Positive Rate');
ylabel('True Positive Rate');
title(sprintf('ROC Curve (AUC = %.4f)', AUC));
grid on;

%% Save trained model
fprintf('\nSaving model...\n');
save('landslide_model.mat', 'net', 'tr', 'hidden_layers', ...
     'accuracy', 'precision', 'recall', 'f1_score', 'AUC');

fprintf('\nAnalysis complete! Results saved to:');
fprintf('\n1. landslide_susceptibility_map.tif');
fprintf('\n2. landslide_model.mat\n');