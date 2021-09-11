function [idxStart, idxEnd] = detectMuscleActivity(emg, options)
fs = options.fs;
minWindowLengthOfMuscleActivity = options.minWindowLengthOfMuscleActivity;
plotSignals = options.plotSignals;
numFreqOfSpec = 50; % The actual number of frequencies is numFreqOfSpec/2
hammingWdwLength = 25;
numSamplesOverlapBetweenWdws = 10;
threshForSumAlongFreqInSpec = options.threshForSumAlongFreqInSpec;
sumEMG = sum(emg, 2);
[spec, dummy, time, ps] = spectrogram(sumEMG, hammingWdwLength, ...
    numSamplesOverlapBetweenWdws, numFreqOfSpec, fs, 'yaxis');
spec = abs(spec);
greaterThanThresh = [0, sum(spec, 1) >= threshForSumAlongFreqInSpec, 0];
diffGreaterThanThresh = abs(diff(greaterThanThresh));
if diffGreaterThanThresh(end) == 1
    diffGreaterThanThresh(end - 1) = 1;
end
diffGreaterThanThresh = diffGreaterThanThresh(1:(end - 1));
idxNonZero = find(diffGreaterThanThresh == 1);
idxOfSamples = floor(time*fs);
numIdxNonZero = length(idxNonZero);
switch numIdxNonZero
    case 0
        idxStart = 1;
        idxEnd = length(sumEMG);
    case 1
        idxStart = idxOfSamples(idxNonZero);
        idxEnd = length(sumEMG);
    otherwise
        idxStart = idxOfSamples(idxNonZero(1));
        idxEnd = idxOfSamples(idxNonZero(end) - 1);
end
% switch numIdxNonZero
%     case 0
%         idxStart = 1;
%         idxEnd = length(sumEMG);
%     case 1
%         idxStart = idxOfSamples(idxNonZero);
%         idxEnd = length(sumEMG);
%     case 2
%         idxStart = idxOfSamples(idxNonZero(1));
%         idxEnd = idxOfSamples(idxNonZero(2) - 1);
%     otherwise
%         diffIdx = idxNonZero(2:end) - idxNonZero(1:(end - 1));
%         [dummy, maxIdx] = max(diffIdx);
%         idxStart = idxOfSamples(idxNonZero(maxIdx));
%         idxEnd = idxOfSamples(idxNonZero(maxIdx + 1) - 1);
% end

% Adding a head and a tail to the detection
numExtraSamples = 25;
idxStart = max(1, idxStart - numExtraSamples);
idxEnd = min(length(sumEMG), idxEnd + numExtraSamples);
if (idxEnd - idxStart) < minWindowLengthOfMuscleActivity
    idxStart = 1;
    idxEnd = length(sumEMG);
end
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
if plotSignals
    figure(1);
    spectrogram(sumEMG, hammingWdwLength, ...
        numSamplesOverlapBetweenWdws, numFreqOfSpec, fs, 'yaxis');
    xlabel('Time (s)');
    ylabel('Frequency (Hz)')
    drawnow;
    
    figure(2);
    subplot(2, 1, 1);
    plot(sumEMG, 'linewidth', 2, 'Color', [0.9 0.7 0.1]);
    subplot(2, 1, 2);
    imagesc(10*log10(sum(spec, 1)));
    colormap jet;
    colorbar;
    axis off;
    drawnow;
end
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
return