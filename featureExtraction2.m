function [testFeatures,featureindices] = featureExtraction2(testData,T,AR_order,level)
% This function is only in support of XpwWaveletMLExample. It may change or
% be removed in a future release.
testFeatures = [];


for idx =1:size(testData,1)
    x1 = testData(idx,:);
    x1 = detrend(x1,0);
    arcoefs = blockAR(x1,AR_order,T);
    se = shannonEntropy(x1,T,level);
    le=leaders(x1,T);
    %[cp,rh] = leaders(x1,T);
    wvar = modwtvar(modwt(x1,'fk8'),'fk8');
    testFeatures = [testFeatures;arcoefs se le wvar']; %#ok<AGROW>

end

featureindices = struct();
% 4*4
featureindices.ARfeatures = 1:16;
startidx = 17;
endidx = 17+(8*4)-1;
featureindices.SEfeatures = startidx:endidx;

startidx = endidx+1;
endidx = startidx+(8*4)-1;
featureindices.logfeatures = startidx:endidx;
%featureindices.HRfeatures = startidx:endidx;
startidx = endidx+1;
endidx = startidx+9;
featureindices.WVARfeatures = startidx:endidx;
end

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
    artmp =  arburg(y(:,kk),order);
    arcfs(:,kk) = artmp(2:end);
end
arcfs = reshape(arcfs,order*numwindows,1);
arcfs = arcfs';
end

function le=leaders(x,numbuffer)
y = buffer(x,numbuffer);
le = zeros(1,size(y,3));
for kk = 1:size(y,3)
      le = wentropy(y(:,kk),'log energy');
  %  [~,h,cptmp] = dwtleader(y(:,kk));
  %  cp(kk) = cptmp(2);
  % rh(kk) = range(h);
end
end
%function [cp,rh] = leaders(x,numbuffer)
%y = buffer(x,numbuffer);
%cp = zeros(1,size(y,2));
%rh = zeros(1,size(y,2));
%for kk = 1:size(y,2)
  %  [~,h,cptmp] = dwtleader(y(:,kk));
  %  cp(kk) = cptmp(2);
  % rh(kk) = range(h);
%end
%end