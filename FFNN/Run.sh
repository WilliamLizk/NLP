#!/bin/bash
#------------------------------------------------------------
#************************************************************
#************************************************************
#**** Desc        : Gen Predict LUcky Num with NLP       ****
#****                                                    ****
#****                                                    ****
#**** Creator     : William Li                           ****
#**** Create Date : 8 Feb 2018                          ****
#************************************************************
#************************************************************
#------------------------------------------------------------
if [[ $# -ne 0 ]]
then
  echo "Usage: $0"
  echo "e.g. $0"
  exit 1
fi

RootPath="$( cd "$( dirname "$0"  )" && pwd  )"
PythonScriptName4Red=${RootPath}/LuckyDrawRed.py
PythonScriptName4Blue=${RootPath}/LuckyDrawBlue.py

echo "The Set 1 Result:"
TrainingDataFile=${RootPath}/Data/RedSet1.csv
python ${PythonScriptName4Red} ${TrainingDataFile}
TrainingDataFile=${RootPath}/Data/BlueSet1.csv
python ${PythonScriptName4Blue} ${TrainingDataFile}

echo "The Set 2 Result:"
TrainingDataFile=${RootPath}/Data/RedSet2.csv
python ${PythonScriptName4Red} ${TrainingDataFile}
TrainingDataFile=${RootPath}/Data/BlueSet2.csv
python ${PythonScriptName4Blue} ${TrainingDataFile}

echo "The Set 3 Result:"
TrainingDataFile=${RootPath}/Data/RedSet3.csv
python ${PythonScriptName4Red} ${TrainingDataFile}
TrainingDataFile=${RootPath}/Data/BlueSet3.csv
python ${PythonScriptName4Blue} ${TrainingDataFile}

echo "The Set 4 Result:"
TrainingDataFile=${RootPath}/Data/RedSet4.csv
python ${PythonScriptName4Red} ${TrainingDataFile}
TrainingDataFile=${RootPath}/Data/BlueSet4.csv
python ${PythonScriptName4Blue} ${TrainingDataFile}


