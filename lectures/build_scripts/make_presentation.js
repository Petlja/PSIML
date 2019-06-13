const fs = require('fs');
const markdownItContainer = require('markdown-it-container')
var markdownItAttrs = require('markdown-it-attrs');
var markdownItMath = require('markdown-it-mathjax');
const  {Marpit} = require('@marp-team/marpit')

// 1. Create instance (with options if you want)
const marpit = new Marpit().use(markdownItContainer, 'container').use(markdownItAttrs).use(markdownItMath)

// 2. Add theme CSS
const theme = `
/* @theme example */

section {
  background-color: white;
  color: black;
  font-size: 30px;
  padding: 40px;
}

h1,
h2 {
  text-align: left;
  margin: 0;
}

h1 {
  color: #8cf;
}

.container {
}

.two_columns {
  display: grid;
  grid-column-gap: 50px;
  grid-template-columns: auto auto;
}

.min_content {
  grid-template-columns: min-content min-content;
}

`
marpit.themeSet.default = marpit.themeSet.add(theme)

console.log("Converting %s to %s", process.argv[2], process.argv[3])

const markdown = fs.readFileSync(process.argv[2], 'utf8');

//// 3. Render markdown
//const markdown = `
//
//# Hello, Marpit!
//
//Marpit is the skinny framework for creating slide deck from Markdown.
//
//---
//
//## Ready to convert into PDF!
//
//You can convert into PDF slide deck through Chrome.
//
//`

const { html, css } = marpit.render(markdown)

// 4. Use output in your HTML
const htmlFile = `
<!DOCTYPE html>
<html>
<head>
<script src="https://cdnjs.cloudflare.com/ajax/libs/mathjax/2.7.3/MathJax.js?config=TeX-MML-AM_CHTML"></script>
</head>
<body>
  <style>${css}</style>
  ${html}
</body></html>
`
fs.writeFileSync(process.argv[3], htmlFile.trim())