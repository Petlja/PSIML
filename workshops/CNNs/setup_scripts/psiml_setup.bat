echo "Setting up workshop, this may take a few minutes..." & ^
git clone https://github.com/Petlja/PSIML.git & ^
cd .\PSIML\workshops\CNNs & ^
conda env create -f environment_gpu.yml & ^
conda activate psiml_gpu & ^
echo "Setup success! Starting jupyter notebook" & ^
jupyter-notebook CNN_WS.ipynb ^ &
cmd /k 

