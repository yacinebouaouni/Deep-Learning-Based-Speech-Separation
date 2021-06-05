clc
[SDR_nmf,~,~,~] = bss_eval_sources(table2array(nmfest).',table2array(speechorig).')
[SDR_dnn,~,~,~] = bss_eval_sources(table2array(speechest).',table2array(speechorig).')

fprintf('NMF  ... SDR = %f\n',SDR_nmf)
fprintf('DNN After 20 Epochs ... SDR = %f\n',SDR_dnn)












