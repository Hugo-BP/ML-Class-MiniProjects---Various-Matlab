clear
clc

% DEFAULT - Patient 1, 70% training 30% test, create classification sets with autoencode level 1 which reduces features to 15

% SETTINGS

%  autoencode_lvl
%  0 = none 
%  1 = 15
%  2 = 10 
%  3 = 3

%  patient
%  1 = 44202 
%  2 = 63502

% percent_train - how much of the dataset is used for training, rest is used for testing

% create_networks_from_scratch - This will override all the old classifiers and create new ones 
% from scratch for both classification and detection if set to anything
% besides -1. Leave as -1  if you dont need new classifiers.

autoencode_lvl = 1; % make sure that if changing this, there are already trained classifiers with this encode level.
patient = 1; 
percent_train = 0.7;    
create_new_networks_from_scratch = -1; % no = -1 | yes = anything else


% USER DOES NOT NEED TO CHANGE ANYTHING PAST THIS POINT


% load and format datasets
% classification
[err_weights_C, T_train_final_C, P_train_final_C, T_test_final_C, P_test_final_C, feature_dim_C] = data_formatting(1, autoencode_lvl, percent_train, patient);
% detection
[err_weights_D, T_train_final_D, P_train_final_D, T_test_final_D, P_test_final_D, feature_dim_D] = data_formatting(2, autoencode_lvl, percent_train, patient);

% create and train classifiers with formatted datasets
if create_new_networks_from_scratch ~= -1
    % train networks for classification
    train_networks(err_weights_C, P_train_final_C, T_train_final_C, feature_dim_C, '_C');
    % train networks for detection
    train_networks(err_weights_D, P_train_final_D, T_train_final_D, feature_dim_D, '_D');
end

% load trained classifiers for both detection and classification
[FF_C, REC_C, LSTM_C, CNN_C, FF_D, REC_D, LSTM_D, CNN_D] = load_networks();

% test all loaded classifiers with P_test
[FF_C_result, REC_C_result, LSTM_C_result, CNN_C_result, FF_D_result, REC_D_result, LSTM_D_result, CNN_D_result] = ...
    simulate_networks( ...
        FF_C, REC_C, LSTM_C, CNN_C,...
        FF_D, REC_D, LSTM_D, CNN_D,...
        P_test_final_C, T_test_final_C, feature_dim_C,...
        P_test_final_D, T_test_final_D, feature_dim_D );

% print metrics through console
display_classifier_result_metrics(FF_C_result, REC_C_result, LSTM_C_result, CNN_C_result, FF_D_result, REC_D_result, LSTM_D_result, CNN_D_result, T_test_final_C, T_test_final_D);






%%%%%%%%%%%%%%%%%%%%%%%%%
%   MAIN METHODS        %
%%%%%%%%%%%%%%%%%%%%%%%%%

function display_classifier_result_metrics(FF_C_result, REC_C_result, LSTM_C_result, CNN_C_result, FF_D_result, REC_D_result, LSTM_D_result, CNN_D_result, T_test_final_C, T_test_final_D)

    disp('METRICS for CLASSIFICATION: ')
    disp(' FF CLASS ')
    calculate_metrics(FF_C_result, T_test_final_C, 1);

    disp(' REC CLASS ')
    calculate_metrics(REC_C_result, T_test_final_C, 1);

    disp(' LSTM CLASS ')
    LSTM_C_result = double(string(LSTM_C_result));
    calculate_metrics(LSTM_C_result, T_test_final_C, 1);

    disp(' CNN CLASS ')
    CNN_C_result = double(string(CNN_C_result));
    calculate_metrics(CNN_C_result, T_test_final_C, 1);

    disp('METRICS for DETECTION: ')
    disp(' FF DET ')
    calculate_metrics(FF_D_result, T_test_final_D, 2);

    disp(' REC DET ')
    calculate_metrics(REC_D_result, T_test_final_D, 2);

    disp(' LSTM DET ')
    LSTM_D_result = double(string(LSTM_D_result));
    calculate_metrics(LSTM_D_result, T_test_final_D, 2);

    disp(' CNN DET ')
    CNN_D_result = double(string(CNN_D_result));
    calculate_metrics(CNN_D_result, T_test_final_D, 2);
end



function [accuracy,sensitivity,specificity] = calculate_metrics(results, expected, class_or_det)
    results = round(results);
    tp = 0;
    tn = 0;
    fp = 0;
    fn = 0;
    accuracy = 0;
    sensitivity = 0;
    specificity = 0;

    % compare expected with results
    acc_correct = 0;
    for i=1 : length(results)
        % accuracy
        if results(i) == expected(i)
            acc_correct = acc_correct+1;
        end

        % classify
        if class_or_det == 1
            if results(i) == 2 && expected(i) == 2
                tp = tp+1;
            elseif results(i) ~= 2 && expected(i) ~= 2
                tn = tn+1;
            elseif results(i) == 2 && expected(i) ~= 2
                fp = fp+1;
            else
                fn = fn+1;
            end

        % detect
        elseif class_or_det == 2
            if results(i) == 1 && expected(i) == 1
                tp = tp+1;
            elseif results(i) ~= 1 && expected(i) ~= 1
                tn = tn+1;
            elseif results(i) == 1 && expected(i) ~= 1
                fp = fp+1;
            else
                fn = fn+1;
            end

        else
            disp('ERROR - METRICS INVALID class_or_det')
            return
        end

    end

    % calculate metrics
    accuracy = acc_correct/length(results) * 100;
    sensitivity = tp/(tp + fn);
    specificity = tn/(tn + fp);

    % display metrics
    fprintf('\n')
    fprintf("TP:%.1f \t FP:%.1f \tTN:%.1f \tFN:%.1f", tp, fp, tn, fn)
    fprintf('\n')
    fprintf('accuracy = %.1f%%\n', accuracy)
    fprintf('Sensivity = %.1f%%\n', sensitivity)
    fprintf('Specificity = %.1f%%\n', specificity)

end



function [FF_C_result, REC_C_result, LSTM_C_result, CNN_C_result, FF_D_result, REC_D_result, LSTM_D_result, CNN_D_result] = simulate_networks(FF_C, REC_C, LSTM_C, CNN_C, FF_D, REC_D, LSTM_D, CNN_D, P_test_final_C, T_test_final_C, feature_dim_C, P_test_final_D, T_test_final_D, feature_dim_D)
	disp('CLASSIFIER TESTING:');

    
    FF_C_result =  normalize_classification_output( sim(FF_C, P_test_final_C) ); 
    disp('  FF_C SUCCESS');
	
    REC_C_result =  normalize_classification_output( sim(REC_C, P_test_final_C) );  
    disp('  REC_C SUCCESS');
    
    P = con2seq(P_test_final_C);
    P = P';
    LSTM_C_result = classify(LSTM_C, P, 'MiniBatchSize', 1000, 'SequenceLength', 'longest');  
    disp('  LSTM_C SUCCESS');

    T = T_convert_output_to_3class(T_test_final_C);
    [P,~] = input_format_CNN(P_test_final_C, T, feature_dim_C);
    CNN_C_result = classify(CNN_C, P, 'MiniBatchSize', 1000, 'SequenceLength', 'longest');  
    disp('  CNN_C SUCCESS');

    disp('SHALLOW SUCCESS');

	FF_D_result =  normalize_detection_output( sim(FF_D, P_test_final_D) );  
    disp('  FF_D SUCCESS');

	REC_D_result =  normalize_detection_output( sim(REC_D, P_test_final_D) );  
    disp('  REC_D SUCCESS');

    P = con2seq(P_test_final_D);
    P = P';
    LSTM_D_result = classify(LSTM_D, P, 'MiniBatchSize', 1000, 'SequenceLength', 'longest');  
    disp('  LSTM_D SUCCESS');

    T = T_convert_output_to_3class(T_test_final_D);
    [P,~] = input_format_CNN(P_test_final_D, T, feature_dim_D);
    CNN_D_result = classify(CNN_D, P, 'MiniBatchSize', 1000, 'SequenceLength', 'longest');   
    disp('  CNN_D SUCCESS');

    disp('DEEP SUCCESS');

end



function [FF_C, REC_C, LSTM_C, CNN_C, FF_D, REC_D, LSTM_D, CNN_D] = load_networks()
    % classification
    FF_C = load('FF_NET_C');
    REC_C = load('REC_NET_C');
    LSTM_C = load('LSTM_NET_C');
    CNN_C = load('CNN_NET_C');

    FF_C = FF_C.trained_net;
    REC_C = REC_C.trained_net;
    LSTM_C = LSTM_C.trained_net;
    CNN_C = CNN_C.trained_net;

    % detection
    FF_D = load('FF_NET_D');
    REC_D = load('REC_NET_D');
    LSTM_D = load('LSTM_NET_D');
    CNN_D = load('CNN_NET_D');

    FF_D = FF_D.trained_net;
    REC_D = REC_D.trained_net;
    LSTM_D = LSTM_D.trained_net;
    CNN_D = CNN_D.trained_net;

    disp('LOAD SUCCESS');

end



function train_networks(err_weights, P, T, feature_dim, type)    
    %%%%%%%%%%%%%%%%%%%%%%%%%
    %    NETWORKS           %
    %%%%%%%%%%%%%%%%%%%%%%%%%
    % set functions - Im using whatever was best for tp2a
    activation_setting_1 = "tansig";
    activation_setting_2 = "purelin";
    training_setting = "trainscg";

    % Feedforward NN
    [trained_net, ~] = net_FeedForward(P, T, activation_setting_1, activation_setting_2, training_setting, err_weights);
    save(strcat('FF_NET', type), 'trained_net'); 
    
    disp('net_FeedForward SUCCESS')

    % Recurrent NN 
    [trained_net, ~] = net_Recurrent(P, T, activation_setting_1, activation_setting_2, training_setting, err_weights);
    save(strcat('REC_NET', type), 'trained_net'); 
    disp('net_Recurrent SUCCESS')

    % LSTM multidimensional time series classification - long short time mem NN
    [trained_net, ~] = deep_net_LSTM(P, T, feature_dim);
    save(strcat('LSTM_NET', type), 'trained_net'); 
    disp('deep_net_LSTM SUCCESS')

    % CNN multiclass classification (deep learning)
    [trained_net, ~] = deep_net_CNN(P, T, feature_dim);
    save(strcat('CNN_NET', type), 'trained_net'); 
    disp('deep_net_CNN SUCCESS')

end



function [err_weights, T_train_final, P_train_final, T_test_final, P_test_final, feature_dimension] = data_formatting(class_or_det, autoencode_lvl, percent_train, patient)
    %%%%%%%%%%%%%%%%%%%%%%%%%
    %   DATA FORMATTING     %
    %%%%%%%%%%%%%%%%%%%%%%%%%
    
    % load patient 1 and 2 data
    [P1_orig, T1_orig, n1, P2_orig, T2_orig, n2] = load_datasets();
    
    % patient select
    if patient == 1
        seizure_number = n1;    
        P_orig = P1_orig; 
        T_orig = T1_orig; 
    elseif patient == 2
        seizure_number = n2;
        P_orig = P2_orig; 
        T_orig = T2_orig; 
    else
        disp('ERROR - invalid patient');
        return
    end
    
    % reduce features
    P_encoded = autoencoder(P_orig, autoencode_lvl); 
    
    % reformat T to pdf class format
    T_formatted = T_convert_to_3_classes_format(T_orig'); 
    
    % get dataset indexes of interest
    [train_index, test_index, seizure_index] = extract_dataset_details(T_formatted, seizure_number, percent_train);
    
    disp('Seizure indexes:')
    disp('       START ---- END')
    disp(seizure_index)
    
    %%%%%%%%%%%%%%%%%%%%%%%%%%
    %     CLASSIFICATION     %
    %%%%%%%%%%%%%%%%%%%%%%%%%%
    if class_or_det == 1
        % get training | test datasets for CLASSIFICATION, P = dataset T = target
        P = P_encoded;  
        T = T_formatted;   
        
        % TRAINING
        P_train_class = P(:,train_index);
        T_train_class = T(:,train_index);
        
        % TESTING
        P_test_class = P(:,test_index);
        T_test_class = T(:,test_index);
        
        % CLASS BALANCING 
        balanced_train_indexes = balance_training_dataset(T_train_class);
        
        % UPDATE TRAIN TEST with BALANCED indexes
        P_train_class = P_train_class(:,balanced_train_indexes);
        T_train_class = T(:,balanced_train_indexes);
        
        % error weights - penalize inter && pre so seizure is more important
        err_weights = penalize_classification(T_train_class); 
        
        % FINAL VARS
        T_train_final = T_convert_to_output_format(T_train_class);
        P_train_final = P_train_class;
        T_test_final = T_convert_to_output_format(T_test_class);
        P_test_final = P_test_class;
    
        save('T_train_final_CLASS', 'T_train_final'); 
        save('P_train_final_CLASS', 'P_train_final'); 
        save('T_test_final_CLASS', 'T_test_final'); 
        save('P_test_final_CLASS', 'P_test_final'); 
    
    %%%%%%%%%%%%%%%%%%%%%%%%%%
    %       DETECTION        %
    %%%%%%%%%%%%%%%%%%%%%%%%%%
    elseif class_or_det == 2
        % get training | test datasets for DETECTION, P = dataset T = target
        P = P_encoded;      
        T = T_formatted; 
        
        % TRAINING
        P_train_det = P(:,train_index);
        T_train_det = T(:,train_index);
        % TESTING
        P_test_det = P(:,test_index);
        T_test_det = T(:,test_index);
        
        % CLASS BALANCING 
        balanced_train_indexes = balance_training_dataset(T_train_det);
        
        % UPDATE TRAIN TEST with BALANCED indexes
        P_train_det = P_train_det(:,balanced_train_indexes);
        T_train_det = T(:,balanced_train_indexes);

        % error weights - penalize inter && pre so seizure is more important
        err_weights = penalize_detection(T_train_det);   
        
        % FINAL VARS
        T_train_final = T_convert_to_output_format(T_train_det);
        P_train_final = P_train_det;
        T_test_final = T_convert_to_output_format(T_test_det);
        P_test_final = P_test_det;

        save('T_train_final_DET', 'T_train_final'); 
        save('P_train_final_DET', 'P_train_final'); 
        save('T_test_final_DET', 'T_test_final'); 
        save('P_test_final_DET', 'P_test_final'); 
    else
        disp('ERROR - invalid class_or_det');
        return
    end 

    % when using static inputSize=29 with encoded data of size 15 for example, net returns error. 
    if autoencode_lvl == 0
        feature_dimension = 29;
    elseif autoencode_lvl == 1
        feature_dimension = 15;
    elseif autoencode_lvl == 2
        feature_dimension = 10;
    elseif autoencode_lvl == 3
        feature_dimension = 3;
    else
        return
        clear %#ok<UNRCH> 
        disp('ERROR - invalid feature_dimension')
    end

    disp('DATA FORMATTING SUCCESS')

end










%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%       NETWORK METHODS
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%%%%%%%%%%%%%%%%%%%%%%%
% Feedforward NN
function [trained_net, tr] = net_FeedForward(P, T, activation_1, activation_2, training, err_weights)
    trained_net = feedforwardnet([100 100]);              
    trained_net.trainFcn = training;          
    trained_net.layers{1}.transferFcn = activation_1;
    trained_net.layers{2}.transferFcn = activation_2;
    
    % window
    trained_net.trainParam.showWindow = 1;
    trained_net.divideFcn = '';
    
    % Parameters
    trained_net.performParam.lr = 0.01; 
    trained_net.trainParam.epochs = 250;  
    trained_net.trainParam.show = 35; 
    trained_net.trainParam.max_fail=500;    
    trained_net.performParam.regularization = tanh(1); 
    trained_net.trainParam.goal = 1e-6;  
    trained_net.performFcn = 'sse';      
    
    % train net gpu
    [trained_net, tr] = train(trained_net, P, T, [], [], err_weights, 'UseParallel', 'yes', 'UseGPU', 'no');

end

%%%%%%%%%%%%%%%%%%%%%%%
% Recurrent NN 
function [trained_net,tr] = net_Recurrent(P, T, activation_1, activation_2, training, err_weights)
    trained_net = layrecnet(1:2,[100 100]);
    trained_net.trainFcn = training;          
    trained_net.layers{1}.transferFcn = activation_1;
    trained_net.layers{2}.transferFcn = activation_2;  
    
    % window
    trained_net.trainParam.showWindow = 1; 
    trained_net.divideFcn = '';
    
    % Parameters
    trained_net.performParam.lr = 0.01;  
    trained_net.trainParam.epochs = 250;     
    trained_net.trainParam.show = 35;        
    trained_net.trainParam.max_fail=500;       
    trained_net.performParam.regularization = tanh(1);  
    trained_net.trainParam.goal = 1e-6;        
    trained_net.performFcn = 'sse';      
    
    % train net gpu
    [trained_net, tr] = train(trained_net, P, T, [], [], err_weights, 'UseParallel', 'yes', 'UseGPU', 'no');

end

%%%%%%%%%%%%%%%%%%%%%%%
% multidimensional time series classification - long short time mem NN
% P needs to be a time series. Could only succeed with con2seq(). timeseries() does not work. also needs to transpose so its (X,1)
% T needs to be categorical. Could not make it work with default T with 3 columns. 
% Japanese example also only used one column with severalsingle digit classes. 
% REFORMAT TARGET FROM OUTPUT FORMAT BACK TO 3-CLASS SINGLE DIGIT FORMAT
function [trained_net, tr] = deep_net_LSTM(P, T, feature_dim)
    % data format for input and target. LSTM is picky about inputs.
    % format to time sequence and transpose
    P = con2seq(P);
    P = P';

    % Revert T from output format to 3Class format. 
    T = T_convert_output_to_3class(T);
    
    % format to categorical and transpose
    T = categorical(T);
    T = T';

    % net settings
    inputSize = feature_dim;  % 29
    output_size = 3; 
    num_hidden = 100;

    maxEpochs = 100;
    miniBatchSize = 1000; 
    
    layers = [ ...
        sequenceInputLayer(inputSize)
        bilstmLayer(num_hidden, 'OutputMode', 'last')
        fullyConnectedLayer(output_size)
        softmaxLayer
        classificationLayer];
    
    options = trainingOptions('adam', ...
        'ExecutionEnvironment', 'cpu', ...
        'GradientThreshold', 1, ...
        'MaxEpochs', maxEpochs, ...
        'MiniBatchSize', miniBatchSize, ...
        'SequenceLength', 'longest', ...
        'Shuffle', 'never', ...
        'Verbose', 0, ...
        'Plots', 'training-progress');
    
    % train & save net
    [trained_net, tr] = trainNetwork(P, T, layers, options);

end

%%%%%%%%%%%%%%%%%%%%%%%
% multiclass classification (deep learning)
function [trained_net, tr] = deep_net_CNN(P, T, feature_dim)
    % Revert T from output format to 3Class format. 
    T = T_convert_output_to_3class(T);

    % function to format data into images -->
    [P,T] = input_format_CNN(P, T, feature_dim);
    
    % format to categorical and transpose
    T = categorical(T);
    T = T';

    % net settings
    output_size = 3;
    inputSize = [feature_dim feature_dim 1]; % 29
    layers = [ ...
        imageInputLayer(inputSize,'Normalization','rescale-zero-one');
        convolution2dLayer(5,20)
        reluLayer
        maxPooling2dLayer(2,'Stride',2)
        fullyConnectedLayer(output_size)
        softmaxLayer
        classificationLayer];
    
    options = trainingOptions('sgdm');
    
    % train & save net
    [trained_net, tr] = trainNetwork(P, T, layers, options);

end

% multiclass classification (deep learning) AUX
% format dataset as img of 29x29
% assume T is a formatted 3class target
% formatted_CNN_input: 29 x 29 x 1 x NumberOfImages
function [P_formatted, T_formatted] = input_format_CNN(P, T, feature_dim)
    [~, y] = size(T);
    % 29x29x1 ||  15x15x1 ||  10x10x1 || 3x3x1
    P_formatted = P(1:feature_dim, 1:feature_dim, 1); 
    T_formatted = ones(1,y);

    T = categorical(T);
    T = T'; 

    % dataset example --> 15x29187 feature_size = 15 --> 15x15 images
    % this means total image should be less than 29187/15 = +/-1900
    % start at second iteration
    num_imgs = 2;
    i = feature_dim+1;
    while i+feature_dim-1 <= size(P,2)
        % only have same class in an image
        count = 0;
        for j=i : i+feature_dim-1
            if T(j) == T(i)
                count = count+1;
            end
        end   

        % update formatted data
        if count == feature_dim && i+feature_dim-1 <= size(P,2) 
            % save P image in the size x size x 1 format
            P_formatted(:, :, :, num_imgs) = P(1 : feature_dim, i : i+feature_dim-1, 1);
            % save T class
            T_formatted(num_imgs) = T(i);

            num_imgs = num_imgs+1;
        end
        i = i + count;

    end 
    
    % round so size of P and T are equal
    if size(T_formatted) ~= num_imgs
        T_formatted = T_formatted(1 : num_imgs-1);
    end

end


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%       DATA FORMAT METHODS   
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%%%%%%%%%%%%%%%%%%%%%%%
% get class balanced training from target
% equilibrate the number of points of the several classes in the
% training set, but not in the testing set. This is the class balancing approach
function [index_training_balanced] = balance_training_dataset(target)
    index_training_balanced = [];

    [~,y] = size(target); 

    seizure_iterations = 0;  
    seizure_pos = [];
    seizure_count = 0;
    
    % seizure indexes
    for i=1 : y           
        % if seizure
        if target(1,i) == 3 
            % check consecutive seizure
            seizure_iterations = seizure_iterations + 1;  

            % if previous is pre
            if target(1,i-1) == 2
                % save start index
                aux(1,1) = i;
                seizure_count = seizure_count + 1; 
            
            % if next not seizure
            elseif target(1,i+1) == 1
                % save end index
                aux(2,1) = i;
                % save start end index
                seizure_pos = [seizure_pos aux]; 
            end
        end
    end    
    
    % go through all seizures, clean excess non-seizure
    limit = 1;
    for i=1 : seizure_count 
        seizure_start = seizure_pos(1,i);
        seizure_end = seizure_pos(2,i);
        
        % consider 15mins prior to seizure
        pre = seizure_start -901;
        
        % avg seizure duration + 15min window
        seizure_length = seizure_pos(2,:) - seizure_pos(1,:) +1; 
        size_sample = seizure_length(i) + 900;   
        
        %  all data from inter to pre  
        population = limit:pre;  
        
        % if non seizure total > size_sample
        if pre - limit +1 > size_sample          
            % get pre pos and seizure indexes, inter is less than sum of these 3
            % https://www.mathworks.com/help/stats/randsample.html
            % For a patient, several interictal points at most equal to the sum of the points of the other classes should be chosen,
            % for example randomly. This corresponds to an under sampling of the interictal phase. 
            index_training_balanced = [ index_training_balanced randsample(population, size_sample) (pre+1 : seizure_end)];
        else
            index_training_balanced = [ index_training_balanced population];
        end
        % update limit for next seizure
        limit = seizure_end +301;
        
    end
    
    % all data from pre to inter
    population = limit : y;      
    if y-limit+1 > size_sample
        index_training_balanced = [ index_training_balanced randsample(population, size_sample) ];
    else
        index_training_balanced = [ index_training_balanced population];
    end
    
end

%%%%%%%%%%%%%%%%%%%%%%%
% train autoencoders to reduce the number of features for the classifiers
function [features] = autoencoder(dataset, number_of_reductions) 
    P = dataset;
    feature_reduction = [15, 10, 3];
    % reduce none
    if(number_of_reductions == 0)
        features = P;
    end
    
    % reduce 1 time
    if (number_of_reductions == 1 || number_of_reductions == 2 || number_of_reductions == 3)
        encoder_output = trainAutoencoder(P, feature_reduction(1), 'MaxEpochs', 50, 'L2WeightRegularization', 0.01, 'SparsityRegularization', 4, 'SparsityProportion', 0.05, 'DecoderTransferFunction', 'purelin');
        features = encode(encoder_output, P);   
    end
    
    % reduce 2 times
    if (number_of_reductions == 2 || number_of_reductions == 3)
        encoder_output = trainAutoencoder(features, feature_reduction(2),'MaxEpochs', 50, 'L2WeightRegularization', 0.01, 'SparsityRegularization', 4, 'SparsityProportion', 0.05, 'DecoderTransferFunction', 'purelin', 'ScaleData', false);
        
        features = encode(encoder_output, features);   
    end
    
    % reduce 3 times
    if (number_of_reductions == 3)
        encoder_output = trainAutoencoder(features, feature_reduction(3),'MaxEpochs', 50, 'L2WeightRegularization', 0.01, 'SparsityRegularization', 4, 'SparsityProportion', 0.05, 'DecoderTransferFunction', 'purelin', 'ScaleData', false);           
        features = encode(encoder_output, features);                  
    end
                
end

%%%%%%%%%%%%%%%%%%%%%%%
% AUX convert T to pdf specifications
% T starts with 2 class only, convert to 3 class format
% ictal == seizure
%{
input:
    0 = non
    1 = seizure
output:
    1 = inter
    2 = pre
    3 = seizure
%}
function [T_final] = T_convert_to_3_classes_format(T_mat)
    [Tx, ~] = size(T_mat);
    for i=1 : Tx
        % if seizure
        if (T_mat(i,1) == 1)
            if (i-900 > 0)
                % convert 600 vals prior to first seizure into 2 (pre) IF NOT 1 || 3
                for j=i-900 : i-1
                    if (T_mat(j,1) ~= 1 || T_mat(j,1) ~= 3)
                        % 0 (non) ---> 2 (pre)
                        T_mat(j,1) = 2; 
                    else % stop for loop
                        break;
                    end
                end 
            else
                % convert first 900 vals into 2 (pre) IF NOT 1 || 3
                for j=1 : i-1 
                    if (T_mat(j,1) ~= 1 || T_mat(j,1) ~= 3)
                        % 0 (non) ---> 2 (pre)
                        T_mat(j,1) = 2;
                    else % stop for loop
                        break;
                    end
                end
            end
            % 1 (seizure) ---> 3 (seizure)
            while T_mat(i,1) == 1
                T_mat(i,1) = 3;
                i = i+1;
            end
        end
    end

    % else if 0 (non) ---> 1 (inter) 
    for i = 1:Tx 
        if T_mat(i,1) == 0 
            T_mat(i,1) = 1;
        end
    end

    % transpose 3 class result 
    T_final = T_mat';
end

% Convert 3 class format to [001] [010] [100] output format 
%{
input:
    1 = inter
    2 = pre
    3 = seizure
output:
    [1 0 0] 
    [0 1 0] 
    [0 0 1] 
%}
function [T_final] = T_convert_to_output_format(T_mat)
    % convert the 3 classes into the [001] format
    [~, y] = size(T_mat);
    T_final = zeros(3,y);

    for i=1 : y
        if(T_mat(1,i) == 1)
            % [1 0 0]
            T_final(1,i) = 1;
        elseif(T_mat(1,i) == 2) 
            % [0 1 0]
            T_final(2,i) = 1; 
        else 
            % [0 0 1]      
            T_final(3,i) = 1;
        end
    end

end

% Convert output format back to 3 class format 
%{
input:
    [1 0 0] 
    [0 1 0] 
    [0 0 1] 
output:
    1
    2
    3
%}
function [T_final] = T_convert_output_to_3class(T_mat)
    [~, columns] = size(T_mat);
    for i=1:columns
        if T_mat(1,i) == 1
            T_final(i) = 1;
        elseif T_mat(2,i) == 1
            T_final(i) = 2;
        elseif T_mat(3,i) == 1
            T_final(i) = 3;
        end
    end
end

%%%%%%%%%%%%%%%%%%%%%%%
% Penalize T so seizure is more important that inter or pre
function [err_weights] = penalize_classification(target)
    [~,y] = size(target);
    err_weights = zeros(1,y);

    for i=1 : y
        if target(1,i) == 1
            % inter weight
            err_weights(1,i) = 0.1;       
    
        elseif target(1,i) == 2
            % pre weight
            err_weights(1,i) = 0.6;                 
    
        else % seizure weight 
            err_weights(1,i) = 0.9;         
        end    
    end

end

%%%%%%%%%%%%%%%%%%%%%%%
% Penalize T so seizure is more important that inter or pre
function [err_weights] = penalize_detection(target)
    [~,y] = size(target);
    err_weights = zeros(1,y);

    for i=1 : y
        if target(1,i) == 0
            % inter weight
            err_weights(1,i) = 0.1;       
        else % seizure weight 
            err_weights(1,i) = 0.9;         
        end    
    end

end


%%%%%%%%%%%%%%%%%%%%%%%
% AUX data - Load datasets
function [dataset_A_feat, dataset_A_trg, n_seizures_A, dataset_B_feat, dataset_B_trg, n_seizures_B] = load_datasets()
    dataset_A = load("44202.mat");
    dataset_B = load("63502.mat");

    n_seizures_A = 22;
    n_seizures_B = 19;
    
    dataset_A_feat = dataset_A.FeatVectSel;
    dataset_A_trg = dataset_A.Trg;

    dataset_B_feat = dataset_B.FeatVectSel;
    dataset_B_trg = dataset_B.Trg;

    dataset_A_feat = dataset_A_feat';
    dataset_A_trg = dataset_A_trg';

    dataset_B_feat = dataset_B_feat';
    dataset_B_trg = dataset_B_trg';    

end

%%%%%%%%%%%%%%%%%%%%%%%
% AUX data - find seizure indexes, divide between train/test
function [train_index, test_index, seizure_index] = extract_dataset_details(dataset, n_seizures, percent_train) %#ok<*AGROW> 
    train_index = [];
    test_index = [];
    training_num_seizures = round(n_seizures * percent_train);
    seizure_index = zeros(n_seizures, 2);
    
    % get [start, end] of each seizure
    j=1;
    is_seizure = false;
    for i=1 : length(dataset)
        % if start of seizure
        if dataset(i) == 3 && is_seizure == false
            seizure_index(j, 1) = i;
            is_seizure = true;
        end
        % if end of seizure
        if dataset(i) == 1 && is_seizure == true
            seizure_index(j, 2) = i;
            is_seizure = false;
            j = j+1;
        end
    end

    % split dataset between train and test using the indexes
    limit = seizure_index(training_num_seizures+1, 2);
    for i=1 : length(dataset)
        if (i <= limit)
            % TRAINING
            train_index = [train_index i]; 
        else
            % TESTING
            test_index = [test_index i];
        end
    end

end

%%%%%%%%%%%%%%%%%%%%%%%
% AUX normalize output simple vector
function [output] = normalize_detection_output(input)
    % turn doubles into either 0 or 1
    for i=1:length(input)
        if input(i) >= 0.5
            output(i) = 1;
        else
            output(i) = 0;
        end
    end
end

% AUX normalize output 3D matrix -> classifier output format
function [output] = normalize_classification_output(input)
    [~,y] = size(input);
    output = zeros(3,y);
    
    % turn doubles into either 0 or 1
    for i=1:y
        if (max(input(:,i)) == input(1,i))
            % inter
            output(1, i) = 1;
            output(2, i) = 0;
            output(3, i) = 0;

        elseif (max(input(:,i)) == input(2,i))
            % pre
            output(1, i) = 0;
            output(2, i) = 1;
            output(3, i) = 0;
            
        elseif (max(input(:,i)) == input(3,i))
            % seizure
            output(1, i) = 0;
            output(2, i) = 0;
            output(3, i) = 1;
        end
    end
end
