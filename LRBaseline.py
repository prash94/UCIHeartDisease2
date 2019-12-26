import sys
sys.path.append('../src')
from ProjectConfig.Config import HDData
from HeartDiseaseProject.notebooks.ModelPipeline import lr_model
HDData1 = HDData[['age','gender','height','weight','ap_hi','ap_lo','cholesterol','gluc','smoke','alco','active']]
target = HDData['cardio']



lr_model(HDData1, target, plty='l2', clswt='balanced')
