% Wine quality validation using knn

train_data= csvread('training_classification_regression_2015.csv');

Xtrain=train_data(1:4000,1:end-2);
Ltrain=train_data(1:4000,end-1);
%disp(Ltrain);
Xvalid=train_data(4001:end,2:end-2);
Lvalid=train_data(4001:end,end-1);

for K=3:10
    [N , ~] = size(Xtrain);
    [Ntest,~] = size(Xvalid);
    distance = zeros(N,Ntest);

    % Calculating the euclidean distance of validation samples from
    % training data set

    for i = 1: Ntest
         for j = 1: N 
            distance(j,i) = norm(Xtest(i,:)-Xtrain(j,:));
         end
    end

    [~,Index]= sort(distance,'ascend');

   %Predicting k-nearest neighbors
    Ltest = zeros(K,Ntest);
    for i = 1:Ntest
        for j=1:K
            Ltest(j,i) = Ltrain(Index(j,i));
        end
    end
    Predicted_labels = round(mode(Ltest));
    Predicted_labels=Predicted_labels';
    
    conf=sum(Predicted_labels==Lvalid)/length(Predicted_labels);
    p=num2str(conf*100);
end

