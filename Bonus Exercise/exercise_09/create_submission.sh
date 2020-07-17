
MODELS_DIR='models/*'
CODE_DIR='exercise_code/'
NOTEBOOKS='*.ipynb'
EXERCISE_ZIP_NAME='exercise_09.zip'
EXERCISE_DIR=$(pwd)

echo 'Zipping file '$EXERCISE_ZIP_NAME
zip -r $EXERCISE_ZIP_NAME $MODELS_DIR $CODE_DIR $NOTEBOOKS
echo $EXERCISE_ZIP_NAME ' created successfully!!! '
