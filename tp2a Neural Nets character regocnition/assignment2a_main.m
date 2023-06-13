% train_setting
% trainlm - best for regression
% trainscg - best for classification and unit column vector targets
% trainrp - best for huge datasets


% learn_setting
% learnp – perceptron rule
% learnpn – normalized perceptron rule
% learngd – gradient rule  -  incremental training
% learngdm – gradient rule improved with momentum  -  incremental training
% learnh – hebb rule (historical)
% learnhd - hebb rule with decaying weight (see help)
% learnwh - Widrow-Hoff learning rule

% activation_setting
% hardlim - binary
% purelin – linear
% logsig – sigmoidal

%start_main()
start_gui()











%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% GUI - lets user choose filter and activation - 3 filters 3 activations = 9 possible combinations --> 9 classifiers need to be created
% gui loads a classifier named classifier_filter_activation, and saves it
% as selected_classifier. This is then loaded in myclassify and used in mpaper.
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function start_gui()
    assignment2a_gui()
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% MAIN - this is where all the functions are
% this is where you create all the classifiers, filters, etc. Where all the
% training is done. The GUI simply loads everything that was created here.
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function start_main() 
% ...
% ...

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% STEP 0 : P matrix - concat 10 matrix of size 50 to create 500 inputs
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
[P, T] = load_PTmats();  % P input T target matrixes


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% STEP 1 : create train validation test target sets
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% split input between train and test
[train_set, test_set] = split_mat(P, 0.7);
% split train further, between train and validation, 85% of 70%
[train_set, validation_set] = split_mat(train_set, 0.85);
% target set - ARIAL.PERFECT
[target_set, comp] = split_mat(T, 0.7);
[target_set, comp] = split_mat(target_set, 0.85);

%[train_set,valInd,test_set] = dividerand(P,0.7,0.15,0.15);
%[target_set,valInd,comp] = dividerand(T,0.7,0.15,0.15);

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% STEP 2 : create filters : Perceptron, Assoc Mem
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% input = mpaper drawings | target = perfect arial | assume size(input) = size(target) 
%perceptron_filter = binary_perceptron(train_set, target_set);

% filter train_set
%filtered_perc = sim(perceptron_filter,train_set);
filtered_ass_memory = associative_memory(train_set, target_set); %#ok<*NASGU> 


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% STEP 3.1 : create and save classifiers : 1layer
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% input = mpaper drawings (filtered / not filtered)
classifier_input_set = train_set; % train_set filtered_perc filtered_ass_memory

% please specify if using a filter or not, used in saving classifier
% no_filter
% perceptron
% associative_memory
filter_setting = 'no_filter';

% settings
activation_setting = 'logsig'; % hardlim purelin logsig
train_setting = 'trainscg';
learn_setting = 'learngdm';

% create target ---> identity matrix * size of input
[~,y] = size(classifier_input_set(1,:));
classifier_target_set = create_classifier_target(y);


% run classifier, auto save with settings as name
% if no filter, then send train_set directly as input
%[classif_one_layer, tr] = one_layer_classifier(classifier_input_set, classifier_target_set, activation_setting, train_setting, learn_setting, filter_setting); 

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% STEP 3.2 : create and save classifiers : 2layer
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% input = mpaper drawings (filtered / not filtered)
classifier_input_set = train_set; % train_set filtered_perc filtered_ass_memory

% please specify if using a filter or not, used in saving classifier
% no_filter
% perceptron
% associative_memory
filter_setting = 'no_filter';

% settings
activation_setting_hidden_layer = 'logsig'; % hardlim purelin logsig
activation_setting_output_layer = 'purelin'; % softmax
train_setting = 'trainscg';
learn_setting = 'learngdm';

% target = identity matrixes * size of input
[~,y] = size(classifier_input_set(1,:));
classifier_target_set = create_classifier_target(y);


% run classifier, auto save with settings as name
% if no filter, then send train_set directly as input
[classif_two_layer, tr] = two_layer_classifier(classifier_input_set, classifier_target_set, activation_setting_hidden_layer, activation_setting_output_layer, train_setting, learn_setting, filter_setting); %#ok<*ASGLU> 

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% STEP 4 : simulate classifier to check if it's good or bad
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% define input (filter || no filter || train set || test set || etc )
sim_input = test_set;
%sim_input = target_set;
%{
% sim classifier
sim_output_one = sim_classifier(classif_one_layer, sim_input);
sim_output_one = normalize_data(sim_output_one);
save("sim_output_one" ,"sim_output_one");


sim_output_two = sim_classifier(classif_two_layer, sim_input);
sim_output_two = normalize_data(sim_output_two);
save("sim_output_two" ,"sim_output_two");

classifier = load('classifier_test.mat');
sim_output_test = sim_classifier(classifier.classifier, sim_input);
sim_output_test = normalize_data(sim_output_test);
save("sim_output_test" ,"sim_output_test");
%}

end


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% AUX FUNCTIONS AND METHODS
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%%%%%%%%%%%%%%%%%%%%%%%
% call mpaper for P mat, duplicate arial for T mat
function create_PTmats_large() %#ok<DEFNU> 
    P = [];
    T = [];
    for c = 1:10
        % write numbers in order horizontal 1 2 3...9 0
        mpaper;
        % load file
        data = load("P.mat");
        % save matrix in var
        mat = data.P; 
        % concat into final matrix
        P = [ P mat ];                                                                       
    end

    % clone arial matrix to be same size as train matrix 50x10 = 500 digits
    for c = 1:50
        T = [ T l.Perfect ]; %#ok<*AGROW> 
    end
    % save final P of size 500
    save('P.mat', 'P')
    save('T.mat', 'T')

end

%%%%%%%%%%%%%%%%%%%%%%%
% load finished mats from folder
function [P, T] = load_PTmats()
   % P = load("P_large.mat"); % size = 500
   % T = load("T_large.mat"); % size = 500   

    P = load("P_very_large.mat"); % size = 1010
    T = load("T_very_large.mat"); % size = 1010
    
    P = P.P;
    T = T.T;

end

%%%%%%%%%%%%%%%%%%%%%%%
% mat splitter script
function [set1, set2] = split_mat(mat, percent)
    [~,n] = size(mat);

    % split matrix by index limit
    lim = round(n*percent); % total size * 0.7
    set1 = mat(:,1:lim); 
    set2 = mat(:,lim+1:n);  

end

%%%%%%%%%%%%%%%%%%%%%%%
% binary perceptron as filter, used in classifier
function [perc, tr] = binary_perceptron(input, target) 
    perc = perceptron();
    perc = configure(perc, input, target);
    % settings
    %perc.inputs{1}.size = 256; % 256 neurons
    perc.layers{1}.size = 256; % 256 neurons
    perc.layers{1}.transferFcn = 'hardlim'; % hardlim neurons
    perc.trainFcn = 'trainr'; % trainc trainr
    perc.inputweights{1,1}.learnFcn = 'learnp';
   
    [perc, tr] = train(perc, input, target);

    save('filter_perceptron','perc'); 

end

%%%%%%%%%%%%%%%%%%%%%%%
% associative memory as filter, used in classifier
function assoc = associative_memory(input, target)
    if size(input) > 750 % transpose, use if mat too big
        assoc = target * input';
    else % pseudo-inverse, use by default
        assoc = target * pinv(input);
    end
    assoc = assoc * input;

    %save('filter_assoc_memory','assoc'); 

end

%%%%%%%%%%%%%%%%%%%%%%%
% create classifier target_set - create an identity matrix and duplicate it until same size as input
function classifier_target_set = create_classifier_target(size_of_input)
    % create a mat of 0s with diagonal of 1s. Size 10
    % digit 1 will be classifier_target_set = 1, digit 2 = 2, etc... digit 0 = 10
    identity_mat = eye(10);
    create_set = [];
    % clone target_set until it has the same size as input
    clone_factor = ceil(size_of_input / 10);
    for c = 1:clone_factor
        create_set = [ create_set identity_mat ];
    end

    classifier_target_set = create_set(:,1:size_of_input);
    %save('classifier_target_set.mat', 'classifier_target_set')

end

%%%%%%%%%%%%%%%%%%%%%%%
% create 1layer classifier - P1 -> AssocMem OR Percep -> P2 -> Classify 
function [classifier,tr] = one_layer_classifier(input, target, activation_setting, train_setting, learn_setting, filter_setting)
% init network - Create a custom neural network. No args =  no inputs, layers or outputs
classifier = network();

% neural network config
classifier.numInputs = 1; 
classifier.inputs{1}.size = 256; % num features 
classifier.numLayers = 1;        % num layers
classifier.layers{1}.size = 10;  % 10 neurons 
classifier.inputConnect(1) = 1;  % connect input to layer 1
classifier.biasConnect(1) = 1;   % connect bias to layer 1
classifier.outputConnect(1) = 1;

% train, learn, activation settings
classifier.trainFcn = train_setting;
classifier.inputweights{1,1}.learnFcn = learn_setting;
classifier.layers{1}.transferFcn = activation_setting;

% settings PDF
W=rand(10,256); % gen matrix 10x256 with random {0,1}
b=rand(10,1);
classifier.IW{1,1}=W;
classifier.b{1,1}= b;
classifier.performParam.lr = 0.1;      % learning rate - 0.01 best so far 0.5 default
classifier.trainParam.epochs = 1000;   % maximum epochs % 1000 default
classifier.trainParam.show = 35;       % show
classifier.trainParam.max_fail=100;    % max number of failures
classifier.trainParam.goal = 1e-6;     % goal=objective, alternative(lower): classifier.trainParam.goal = 1e-10; 
classifier.performFcn = 'sse';         % criterion


% train classifier
[classifier, tr] = train(classifier, input, target);

% save classifier in a file
filename =  strcat('classifier_with_', filter_setting, '_', activation_setting, '_', train_setting, '_', learn_setting);
save(filename,'classifier'); 

end

%%%%%%%%%%%%%%%%%%%%%%%
% create 2layer classifier - Same as 1layer but with hidden layer that has +10 neurons
function [classifier,tr] = two_layer_classifier(input, target, activation_setting_hidden_layer, activation_setting_output_layer, train_setting, learn_setting, filter_setting)  
% define hidden layer size, more than 10
hidden_layer_size = 20;  

% innit network - Two Layers: 1 hidden, 1 output, creates 2 layers by default. 
classifier = feedforwardnet(hidden_layer_size);       

% neural network config
% ...

% train, learn, activation settings
classifier.trainFcn = train_setting; 
classifier.inputweights{1,1}.learnFcn = learn_setting;
classifier.layers{1}.transferFcn = activation_setting_hidden_layer;            
classifier.layers{2}.transferFcn = activation_setting_output_layer; 
  
% window settings
classifier.trainParam.showWindow = 1;   
classifier.divideFcn = 'dividetrain';

% settings PDF
classifier.performParam.lr = 0.1;      % learning rate - 0.01 best so far 0.5 default
classifier.trainParam.epochs = 1000;   % maximum epochs % 1000 default
classifier.trainParam.show = 35;       % show
classifier.trainParam.max_fail=100;    % max number of failures
classifier.trainParam.goal = 1e-6;     % goal=objective, alternative(lower): classifier.trainParam.goal = 1e-10; 
classifier.performFcn = 'sse';         % criterion


% train classifier
[classifier, tr] = train(classifier, input, target);

% save classifier in a file
filename =  strcat('classifier_with_', filter_setting, '_', activation_setting_hidden_layer, '_', activation_setting_output_layer, '_', train_setting, '_', learn_setting);
save(filename,'classifier'); 

end

% normalize data ex 0.5111, 0.4991, etc into [0,1]
% used in myclassify and non linear classifiers
function [output] = normalize_data(input)
    [x,y] = size(input);
    
    for i=1:y
        m = max(input);
        for j=1:x
            if input(j,i) ~= m(i)
                input(j,i) = 0;
            else
                input(j,i) = 1;
            end
        end
    end
    % return normalized data
    output = input;

end

%%%%%%%%%%%%%%%%%%%%%%%
% simulate classifier with input 
function classifier_output = sim_classifier(classifier, input)
    classifier_output = sim(classifier, input);
end





