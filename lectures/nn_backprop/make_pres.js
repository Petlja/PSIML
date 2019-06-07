import fs from 'fs'
import Marpit from '@marp-team/marpit'

const markdownItContainer = require('markdown-it-container')

var markdownItAttrs = require('markdown-it-attrs');

// 1. Create instance (with options if you want)
const marpit = new Marpit().use(markdownItContainer, 'two_columns').use(markdownItContainer, 'first_column').use(markdownItContainer, 'second_column').use(markdownItAttrs)

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

.two_columns {
  display: grid;
  grid-column-gap: 50px;
  grid-template-columns: auto auto;
}

`
marpit.themeSet.default = marpit.themeSet.add(theme)

const markdown = fs.readFileSync('nn_backprop.md', 'utf8');

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
<html><body>
  <style>${css}</style>
  ${html}
</body></html>
`
fs.writeFileSync('nn_backprop.html', htmlFile.trim())