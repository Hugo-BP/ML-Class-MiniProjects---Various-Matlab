%%%%%%%%%%%%%%%%%%%%%%%
% myclassify is used in OCR_FUN()
function [predicted] = myclassify(d, ~)                            
    disp('classifier')
    classifier = load('selected_classifier.mat');

    % sim(finished_network, mpaper_dataset)
    simulate = sim(classifier.classifier, d);
    disp('simulate')
    % normalize doubles into [0,1]
    norm = normalize_data(simulate);
    disp('norm')
    [x, y] = size(norm);
    %print(size(norm))
    
    predicted = ones(y,1);
    predicted(1:y,1) = -1;
    disp('predicted')
    % fill predicted numbers with nomralized sim out
    for i=1:y
        for j=1:x
            if norm(j,i) == 1
                predicted(i,1) = j;
                break
            end
        end
    end
    disp('end')
    
    a = [ 1 2 3 4 5 6 7 8 9 10 ];
    T = [];
    for c = 1:5
        T = [ T a ]; %#ok<*AGROW> 
    end

    % print correct
    p = predicted;
    p = reshape(p.', 1, []);
    T = reshape(T.', 1, []);

    h = 0 ;
    for i=1:50
        if p(i) == T(i)
            h = h+1;
        end
    end

    c = h/50 * 100 %#ok<*NASGU,NOPRT> 
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
