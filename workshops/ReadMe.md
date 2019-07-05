# PSIML Workshops

This directory contains source code for workshops used during PSIML seminar.

## Prerequisites

To work with workshops code one needs to have:
* Windows:
  * Visual Studio 2017
  * Python tools for Visual Studio
  * Python installation
* Linux:
  * Python
  * make
  * Code editor (Visual Studio code is recommended)

## Setting up the environment
1. Install miniconda for Python 3.7 from e.g. [here](https://docs.conda.io/en/latest/miniconda.html)
2. Use conda environment made out of *environment.yml* file in *workshops* directory.
   Assuming *<clone_dir>* is the directory where you cloned this repo, the command to make environment is:
  ```
  conda env create -f <clone_dir>\workshops\environment.yml -p <environment_name>
  ```
3. Activate the conda environment:
  ```
  conda activate <environment_name>
  ```
4. Build the solution. Instructions for building differ based on the operating system, as follows.
  * *Windows*:
    Edit *<clone_dir>\workshops\common.props*, where *<clone_dir>* is the directory where you cloned this repo, so that *PythonPath* points to the directory that contains python.exe that you want to use.
    *PythonPath* should end with a trailing backslash.
    Start Developer Command Prompt for Visual Studio 2017 (can be found using Windows search).
    cd to *workshops* directory and issue following command:
    ```
    msbuild /p:Platform="Any CPU";Configuration=Debug workshops.sln
    ```
    Note: if build failed with "Python.exe not found" please check that your *PythonPath* is set correctly, as explained above.
  * *Linux*:
  
    Invoke make from *workshops* directory in terminal:
    ```
    make
    ```
5. Once build process is finished, Jupyter notebooks are generated inside *build* directory in *workshops*. Jupyter notebooks can then be started in Anaconda by cd-ing to that directory and issuing:
    ```
    jupyter notebook
    ```
    from terminal. If all went well, jupyter notebook will open in browser.


## Code organisation rules

* Each workshop is in the separate directory
* Each workshop is implemented in python
* Each workshop implementation needs to be convertible to Jupyter notebook (more on that below)

## How to add new workshop

To add new workshop follow the steps:
* Create new directory in the *workshops* folder
* Add necessary python files to that folder (preferably one for now)
* *Windows specific*:
  * Add Visual Studio python project. Easiest way to do this is to copy and rename some existing project and then edit it in text editor. Fields that need to be changed are ProjectGuid (new random GUID needs to be added) and list of files (`<compile Include="file.py"/>` elements). Python files that are to be converted to notebook need to be marked inside project file by setting *Pynb* property on that file like in the example below:

    ```
    <Compile Include="file.py">
        <Pynb>true</Pynb>
    </Compile>
    ```
  * Once project is created it needs to be added to Visual Studio solution workshops.sln in *workshops* directory. Don't forget to add build configurations for the new project as well inside the ProjectConfigurationPlatforms section.
* *Linux specific*:
  * Add workshop Makefile. Easiest way to do this is to copy existing workshop makefile and then edit it in text editor. Relevant macros whose values need to be changed are:
    * NOTEBOOK_SOURCE_FILES - Python source files to be converted to jupyter notebook.
    * OTHER_SOURCE_FILES - Other python source files.
    * RESOURCE_DIRECTORIES - List of directories containing additional resources (models, data, images etc.)
  * Add new workshop to master Makefile (inside *workshops* directory) by appending its name to WORKSHOPS macro (space separated).

Additionally, for the conversion to make sense, file author needs to add Jupyter markers in the code. Supported markers are `# <codecell>` for the new code cell and `# <markdowncell>` for markdown cell.
Displaying imported python files is currently not supported by Jupyter, so if this is important authors are encouraged to avoid using imports and implement code in the single file. Issue with importing the files will be addressed in the future.

To test if Jupyter conversion works one needs to build workshops. Follow the steps from above to build and test jupyter conversion.
