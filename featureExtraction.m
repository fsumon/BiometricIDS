%function [trainFeatures, testFeatures,featureIndices] = featureExtraction(trainData,testData,T,AROrder,level)
%function [allFeatures,featureIndices] = featureExtraction(TrainTestData,T,AROrder,level)


function [testFeatures,featureindices] = featureExtraction(testData1,T,AROrder,level)


%function [trainFeatures, testFeatures,featureIndices] = featureExtraction(trainData,testData,T,level)
testFeatures = [];
for idx =1:size(testData1,1)
    x1 = testData1(idx,:);
    x1 = detrend(x1,0);
    arcoefs = blockAR(x1,AROrder,T);
    se = shannonEntropy(x1,T,level);
    %[cp,rh] = leaders(x1,T);
    wvar = modwtvar(modwt(x1,'haar'),'haar');
    sep = pentropy(x1,T);
    ifq=  instfreq(x1,T);
   % testFeatures = [testFeatures;arcoefs se wvar' sep'];
    testFeatures = [testFeatures;arcoefs se wvar'];
    %testFeatures = [testFeatures; wvar'];%#ok<AGROW>

end

%allFeatures=featureExtract(TrainTestData, T, AROrder, level);
%trainFeatures = featureExtract(trainData, T, level);
%testFeatures = featureExtract(testData, T, level);
featureindices = struct();
% 4*8
featureindices.ARfeatures = 1:4;
startidx = 5;
endidx = 5+(1*1)-1;
featureindices.SEfeatures = startidx:endidx;
startidx = endidx+1;
endidx = startidx+12;
featureindices.WVARfeatures = startidx:endidx;
end

%%%function [features] = featureExtract(data, T, AROrder,level)
%function [features] = featureExtract(data, T, level) features = [];
  %  for i =1:size(data,1)
    
       %  x = data(i,:);
       %  x = detrend(x,0);
      %   y = buffer(x,T);
       %  windows = round(numel(x) / T);
       %  
       %  length = size(y,2);
    
       %  ARCoeffient = zeros(AROrder,length);
        % entropy = zeros(2^level,length);
        %pulse = zeros(1,length);
        %rate = zeros(1,length);
    
        %j = 1;
        % j = 1;
        % while j < length + 1
            % temp =  arburg(y(:,j),AROrder);
            % ARCoeffient(:,j) = temp(2:end);
            
            % packetTransform = modwpt(y(:,j),level);
            %t = sum(packetTransform.^2,2);
             % t = sum(packetTransform.^2,2);
             %z = packetTransform.^2./t;
             % z = packetTransform.^2./t;
             %entropy(:,j) = -sum(z.*log(z+eps),2);
            
           % [~,h,tmp] = dwtleader(y(:,j));
            %pulse(j) = tmp(2);
            %rate(j) = range(h);
            % j = j + 1;
       % end
    
  
        %%ARCoeffient = reshape(ARCoeffient,AROrder * windows,1);
        %entropy = reshape(entropy,2^level * windows,1);
        %variance = modwtvar(modwt(x,'db2'),'db2');
        
        %features = [features; ARCoeffient' entropy' pulse rate variance'];
        %features = [features; ARCoeffient' entropy' variance'];
        %features = [features; ARCoeffient' variance'];
        %features = [features; entropy' variance'];   
   %%% end%%%

 %end
 
 
function se = shannonEntropy(x,numbuffer,level)
numwindows = round(numel(x)/numbuffer);
y = buffer(x,numbuffer);
se = zeros(2^level,size(y,2));
for kk = 1:size(y,2)
    wpt = modwpt(y(:,kk),level);
    % Sum across time
    E = sum(wpt.^2,2);
    Pij = wpt.^2./E;
    % The following is eps(1)
    se(:,kk) = -sum(Pij.*log(Pij+eps),2);
end
se = reshape(se,2^level*numwindows,1);
se = se';
end


function arcfs = blockAR(x,order,numbuffer)
numwindows = round(numel(x)/numbuffer);
y = buffer(x,numbuffer);
arcfs = zeros(order,size(y,2));
for kk = 1:size(y,2)
    %artmp =  arburg(y(:,kk),order);
    artmp =  armcov(y(:,kk),order);
    arcfs(:,kk) = artmp(2:end);
end
arcfs = reshape(arcfs,order*numwindows,1);
arcfs = arcfs';
end