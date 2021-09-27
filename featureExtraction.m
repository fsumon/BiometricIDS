%function [trainFeatures,featureindices] = featureExtraction(trainData,T,AR_order,level)
function [trainFeatures,featureindices] = featureExtraction(trainData,T,AR_order,level)
% This function is only in support of XpwWaveletMLExample. It may change or
% be removed in a future release.
trainFeatures = [];


for idx =1:size(trainData,1)
    x = trainData(idx,:);
    x = detrend(x,0);
    arcoefs = blockAR(x,AR_order,T);
    se = shannonEntropy(x,T,level);
    %le=leaders(x,T);

    % [cp,rh] = leaders(x,T);
 
      wvar = modwtvar(modwt(x,'haar'),'haar'); % 85.5%
      
      sigenergy = sum(x.^2,2);
      absvalue=abs(x);
      %wvar1 = modwtvar(modwpt(x,'haar'),'haar'); % 85.5%
      %wvar2 = modwtvar(modwt(x,'haar'),'haar'); % 85.5%
     
   % wvar = modwtvar(modwt(y,'db15'),'db15'); % 82%
   % wvar = modwtvar(modwt(y,'db4'),'db4');
    
    %wvar = modwtvar(modwt(y,'coif1'),'coif1');
    %wvar = modwtvar(modwt(y,'fk22'),'fk22');
    %wvar = modwtvar(modwt(x,'sym2'),'sym2');
    sep = pentropy(x,T);
    ifq=  ceil(instfreq(x,T));
    ps= ceil(pspectrum(x,T));
    wp = wpe(x,T,level);
    
   
    %cvar=cwt(x,'bump')
    wtropy = wentropy(x,'shannon');
    wtropy1 = wentropy(x,'log energy');
    
    %wavelet packet energy
    
    
    [C,L] = wavedec(x,12,'haar');
    [Ea,Ed] = wenergy(C,L);
    
    %trainFeatures = [trainFeatures; arcoefs se wvar']; %#ok<AGROW>
    %trainFeatures = [trainFeatures; wvar' sigenergy' wtropy wtropy1 se' Ea Ed];
     %trainFeatures = [trainFeatures; wvar' Ea Ed sigenergy'];
    %trainFeatures = [trainFeatures; wvar' sigenergy' Ed wp wtropy];
    %trainFeatures = [trainFeatures; arcoefs se wvar' wtropy];
     trainFeatures = [trainFeatures; arcoefs se wvar'];
    %trainFeatures = [trainFeatures; absvalue] ;
end

featureindices = struct();
featureindices.SpectralEntropy = 1:32;
startidx = 33;
endidx = startidx+11;
featureindices.WVARfeatures = startidx:endidx;
end


function wp = wpe(x,numbuffer,level)
numwindows = round(numel(x)/numbuffer);
y = buffer(x,numbuffer);
wp = zeros(2^level,size(y,2));
for kk = 1:size(y,2)
    wpt = modwpt(y(:,kk),level);
    wp(:,kk) = sum(wpt.^2,2);
end
wp = reshape(wp,2^level*numwindows,1);
wp = wp';
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