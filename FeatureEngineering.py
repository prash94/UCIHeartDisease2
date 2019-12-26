from HeartDiseaseProject.notebooks.FeatureEngineeringPipeline import imputer

import sys
sys.path.append('../src')
from ProjectConfig.Config import HDData

imputer(HDData,'median')
print(HDData['age'].isnull().sum())

