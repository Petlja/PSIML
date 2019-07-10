## Camp lectures
this directory contains source code for some of the PSI:ML lectures in format suitable for git versioning (MARP, latex)

# Markdown based slides

The markdown based slides are based on marpit framework which is in turn based on pure markdown with extensions.

## Prerequisites

* Visual studio 2017 (or MSbuild at least)
* Google chrome (for converting html to pdf)
* Node.js
  * Node.js can be installed from https://nodejs.org/en/download. Windows zip binary x64 should be downloaded and extracted to directory of choice (in rest of the document will be referred to as _nodejs_root_).
* Following node.js packages should be installed globally:
  * @marp-team/marpit@1.1.0
  * markdown-it-attrs@2.4.1
  * markdown-it-container@2.0.0
  * markdown-it-mathjax@2.0.0
* Node.js packages are installed globally using following command (replace package with actual package name):
```
npm install -g --save package
```


## Presentation organization rules

* Each presentation is in the separate directory
* Each presentation is implemented in markdown
* Each presentation uses markdown with predefiend set of extensions.
   * Current list of extensions are:
     * @marp-team/marpit (for converting markdown to slides)
     * markdown-it-attrs (to enable setting style on markdown element)
     * markdown-it-container (to enable custom elements in markdown)
     * markdown-it-mathjax (to enable math rendering, supports latex and asciimath)

## Add new presentation

To add new presentation follow the steps:
* Create new directory in the _lectures_ folder named after presentation (referred to below as _new_pres_)
* Add markdown file _new_pres_.md
* Add any additional content (images etc. inside _media_ folder)
* Create visual studio presentation project

### Visual studio presentation project

Visual studio presentation project should be called _new_pres_.presproj. It should be added to _lectures.sln_ (similar to existing projects, just refer to them).

Typical presproj file is given below:

```
<?xml version="1.0" encoding="utf-8"?>
<Project ToolsVersion="4.0" xmlns="http://schemas.microsoft.com/developer/msbuild/2003">
  <PropertyGroup>
    <SchemaVersion>2.0</SchemaVersion>
    <ProjectGuid>{70D37464-E2B9-441E-AE9A-32C82B1AC35A}</ProjectGuid>
  </PropertyGroup>
  <Import Project="$(SolutionDir)\common.props" />
  <ItemGroup>
    <PresentationItem Include="new_pres.md"/>
    <AuxItem Include="media\image.png"/>
  </ItemGroup>
  <Import Project="$(SolutionDir)\common.targets" />
</Project>
```

Presproj file must have exactly one PresentationItem declared (presentation markdown file) and arbitrary number of AuxItems (images etc.).

Once presentation is written final output can be obtained by building in Visual Studio. It places final presentation at lectures\build\presentations\new_pres directory. To be able to build presentation one needs to add path to node.js and chrome.exe. It is done by creating _common.user.props_ file side by side with _common.props_ in _lectures_ directory with the following content (replace NodeJsPath text with _nodejs_root_ from prerequisites section):

```
<?xml version="1.0" encoding="utf-8"?>
<Project InitialTarget="CheckPython" ToolsVersion="4.0" xmlns="http://schemas.microsoft.com/developer/msbuild/2003">
  <PropertyGroup Label="UserMacros">
    <NodeJsPath>F:\Programs\node.js\node-v10.16.0-win-x64\</NodeJsPath>
    <ChromeExePath>C:\Program Files (x86)\Google\Chrome\Application\chrome.exe</ChromeExePath>
  </PropertyGroup>
</Project>
```

# Markdown and extensions

For writing presentation one can use usual markdown features (https://github.com/adam-p/markdown-here/wiki/Markdown-Cheatsheet). Slides are delimited by using three successive minus signs. Example is given below:

```
---

# Neural Networks and Backpropagation

---

# Agenda

* Neural networks
* Backpropagation
* Best practices
* Workshop

---
```

TODO: Add documentation for installed extensions and their usage.