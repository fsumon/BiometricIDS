function [trainFeatures, testFeatures,featureIndices] = featureExtraction(trainData,testData,T,AROrder,level)

trainFeatures = featureExtract(trainData, T, AROrder, level);
testFeatures = featureExtract(testData, T, AROrder, level);

featureIndices = struct();
featureIndices.ARFeatures = 1:32;
%featureIndices.ARFeatures = 1:40;
start = 33;
%start = 41;
%ending = 33+(16*8)-1; %ECG
%ending = 41+(16*10)-1; %ECG
%ending = 41+(16*20)-1; %PPG
ending = 33+(2*80)-1; %ECGPPG
featureIndices.entropyFeatures = start:ending;

%start = ending+1;
%ending = start+7;
%featureIndices.pulseFeatures = start:ending;

%start = ending+1;
%ending = start+7;
%featureIndices.rateFeatures = start:ending;

start = ending+1;
%ending = start+13;
ending = start+11;
featureIndices.varianceFeatures = start:ending;

end

function [features] = featureExtract(data, T, AROrder, level)
    features = [];
    for i =1:size(data,1)
    
        x = data(i,:);
        x = detrend(x,0);
        y = buffer(x,T);
        windows = round(numel(x) / T);
       
        length = size(y,2);
    
        ARCoeffient = zeros(AROrder,length);
        entropy = zeros(2^level,length);
        %pulse = zeros(1,length);
        %rate = zeros(1,length);
    
        j = 1;
        while j < length + 1
            temp =  arburg(y(:,j),AROrder);
            ARCoeffient(:,j) = temp(2:end);
            
            packetTransform = modwpt(y(:,j),level);
            t = sum(packetTransform.^2,2);
            z = packetTransform.^2./t;
            entropy(:,j) = -sum(z.*log(z+eps),2);
            
            [~,h,tmp] = dwtleader(y(:,j));
            %pulse(j) = tmp(2);
            %rate(j) = range(h);
            j = j + 1;
        end
    
        ARCoeffient = reshape(ARCoeffient,AROrder * windows,1);
        entropy = reshape(entropy,2^level * windows,1);
        variance = modwtvar(modwt(x,'db2'),'db2');
        
        %features = [features; ARCoeffient' entropy' pulse rate variance'];
        features = [features; ARCoeffient' entropy' variance'];   
    end

end