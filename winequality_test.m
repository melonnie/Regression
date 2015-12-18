% Wine quality test using knn

train_data= csvread('training_classification_regression_2015.csv');
test_data =csvread('challenge_public_test_classification_regression_2015.csv');

Xtrain=train_data(:,1:end-2);
Ltrain=train_data(:,end-1);

Xtest=test_data(:,2:end-2);

K=4;

[N , ~] = size(Xtrain);
[Ntest,~] = size(Xtest);
distance = zeros(N,Ntest);

% Calculating the euclidean distance of validation samples from
% training data set
for i = 1: Ntest
     for j = 1: N 
        distance(j,i) = norm(Xtest(i,:)-Xtrain(j,:));
     end
end

[~,Index]= sort(distance,'ascend');

%calculate k-nearest neighbors
Ltest = zeros(K,Ntest);
for i = 1:Ntest
    for j=1:K
        Ltest(j,i) = Ltrain(Index(j,i));
    end
end

test_Predicted_labels = round(mode(Ltest));
test_Predicted_labels = test_Predicted_labels';

ids=test_data(:,1);
final = [ids test_Predicted_labels];
csvwrite('output_type_k4.csv',final,'');


