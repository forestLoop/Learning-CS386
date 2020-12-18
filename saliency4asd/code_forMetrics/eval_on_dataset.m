function eval_on_dataset()

fileID = fopen('eval_result.txt','w');

trueSaliencyFolder = "C:\MyCode\DIP\DIP2\TrainingDataset\TrainingData\ASD_FixMaps\";
trueFixationFolder = "C:\MyCode\DIP\DIP2\TrainingDataset\AdditionalData\ASD_FixPts\";
predSaliencyFolder = "C:\MyCode\DIP\DIP2\ImagesTest_Result\";

SIM_total = 0;
CC_total = 0;
KL_total = 0;
NSS_total = 0;
AUC_J_total = 0;
AUC_B_total = 0;

predSaliencyFiles = dir(fullfile(predSaliencyFolder, "*.png"));
for idx = 1:length(predSaliencyFiles)
    fileName = predSaliencyFiles(idx).name;
    fprintf(fileID, "--- %s ---\n", fileName);
    predSaliencyMap = imread(fullfile(predSaliencyFolder, fileName));
    temp = split(fileName, ".");
    fileIdx = temp(1);
    trueSaliencyMap = imread(fullfile(trueSaliencyFolder, fileIdx + "_s.png"));
    trueFixationMap = imread(fullfile(trueFixationFolder, fileIdx + "_f.png"));
    % fprintf("%s\n%s\n%s\n", fullfile(predSaliencyFolder, fileName),fullfile(trueSaliencyFolder, fileIdx + "_s.png"), fullfile(trueFixationFolder, fileIdx + "_f.png"))
    SIM_val = similarity(predSaliencyMap, trueSaliencyMap);
    SIM_total = SIM_total + SIM_val;
    CC_val = CC(predSaliencyMap, trueSaliencyMap);
    CC_total = CC_total + CC_val;
    KL_val = KLdiv(predSaliencyMap, trueSaliencyMap);
    KL_total = KL_total + KL_val;
    NSS_val = NSS(predSaliencyMap, trueFixationMap);
    NSS_total = NSS_total + NSS_val;
    AUC_J_val = AUC_Judd(predSaliencyMap, trueFixationMap,0);
    AUC_J_total = AUC_J_total + AUC_J_val;
    AUC_B_val = AUC_Borji(predSaliencyMap, trueFixationMap);
    AUC_B_total = AUC_B_total + AUC_B_val;
    fprintf(fileID, "SIM: %.3f\nCC: %.3f\nKL: %.3f\nNSS: %.3f\nAUC-J: %.3f\nAUC-B: %.3f\n", SIM_val, CC_val, KL_val, NSS_val, AUC_J_val, AUC_B_val);
end

fprintf(fileID, "=== Summary ===\n");
fprintf(fileID, "SIM: %.3f\n", SIM_total / length(predSaliencyFiles));
fprintf(fileID, "CC: %.3f\n", CC_total / length(predSaliencyFiles));
fprintf(fileID, "KL: %.3f\n", KL_total / length(predSaliencyFiles));
fprintf(fileID, "NSS: %.3f\n", NSS_total / length(predSaliencyFiles));
fprintf(fileID, "AUC-J: %.3f\n", AUC_J_total / length(predSaliencyFiles));
fprintf(fileID, "AUC-B: %.3f\n", AUC_B_total / length(predSaliencyFiles));
fprintf(fileID, "===============\n");
