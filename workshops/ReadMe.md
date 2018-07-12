# PSIML Workshops

This directory contains source code for workshops used during PSIML seminar.

## Prerequisites

To work with workshops code one needs to have:
* Visual Studio 2017
* Python tools for Visual Studio
* Python installation

## Code organisation rules

* Each workshop is in the separate directory
* Each workshop is implemented in python
* Each workshop implementation needs to be convertible to Jupyter notebook (more on that below)

## How to add new workshop

To add new workshop follow the steps:
* Create new directory in the *workshop* folder
* Add necessary python files to that folder (preferably one for now)
* Add Visual Studio python project. Easiest way to dot his is to copy and rename some existing project and then edit it in text editor. Fields that need to be changed are ProjectGuid (new random GUID needs to be added) and list of files (`<compile Include="file.py"/>` elements).
* Once project is created it needs to be added to Visual Studio solution workshops.sln in *workshop* directory.

Once workshop is added it needs to be made convertible to Jupyter notebook. Python files that are to be converted to notebook need to be marked inside project file by setting *Pynb* property on that file like in the example below:

```
<Compile Include="file.py">
    <Pynb>true</Pynb>
</Compile>
```
Additionally, for the conversion to make sense, file author needs to add Jupyter markers in the code. Supported markers are `# <codecell>` for the new code cell and `# <markdowncell>` for markdown cell.
Displaying imported python files is currently not supported by Jupyter, so if this is important authors are encouraged to avoid using imports and implement code in the single file. Issue with importing the files will be addressed in the future.

To test if Jupyter conversion works one needs to build solution. Start Visual Studio command prompt, cd to *workshop* directory and issue following command:
```
msbuild.exe /t:build workshops.sln
```
Once build process is finished, Jupyter notebooks are generated inside *build* directory in *workshop*. Jupyter notebooks can then be started by cd-ing to that directory and issuing:
```
jupyter notebook
```
from command prompt. If all went well, jupyter notebook will open in browser.
