function [pureGesture, idxStart, idxEnd] = segmentationGesture(rawGesture)

% parámetros de la segmentación (detectMuscleActivity)
options.fs = 200; % Sampling frequency of the emg
options.minWindowLengthOfMuscleActivity = 80;
options.threshForSumAlongFreqInSpec = 10; % Threshold for detecting the muscle activity
options.plotSignals = false;

%% preprocesamiento
freqFiltro = 0.05;
[Fb, Fa] = butter(4, freqFiltro, 'low'); % creando filtro
filteredEmg = filtfilt(Fb, Fa, abs(rawGesture));

%% segmentación
[idxStart, idxEnd] = detectMuscleActivity(filteredEmg, options);

%% recorte señal
pureGesture = rawGesture(idxStart:idxEnd, :);
end
