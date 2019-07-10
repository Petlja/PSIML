// Load filesytem related modules.
const fs = require('fs');
const path = require('path');

// Additional markdown extensions used.
const markdownItContainer = require('markdown-it-container')
var markdownItAttrs = require('markdown-it-attrs');
var markdownItMath = require('markdown-it-mathjax')({
    beforeInlineMath: '\\(',
    afterInlineMath: '\\)'
});
const {Marpit} = require('@marp-team/marpit')

// Create marpit instance (using loaded markdown extensions).
const marpit = new Marpit().use(markdownItContainer, 'container').use(markdownItAttrs).use(markdownItMath)

// Load default CSS theme.
const theme = fs.readFileSync(path.resolve(__dirname, 'default.css'), 'utf8');

// TODO: Here we should probably load presentaiton specific css and concatenate.

// Set default marp css theme.
marpit.themeSet.default = marpit.themeSet.add(theme)

console.log("Converting %s to %s...", process.argv[2], process.argv[3])

// Load presentation file.
const markdown = fs.readFileSync(process.argv[2], 'utf8');

// Render markdown using marpit.
const { html, css } = marpit.render(markdown)

// Construct output HTML.
const htmlFile = `
<!DOCTYPE html>
<html>
<head>
<script src="https://cdnjs.cloudflare.com/ajax/libs/mathjax/2.7.5/MathJax.js?config=TeX-MML-AM_CHTML"></script>
</head>
<body>
  <style>${css}</style>
  ${html}
</body></html>
`

// Save final HTML to file.
fs.writeFileSync(process.argv[3], htmlFile.trim())

console.log("Converting %s to %s finished successfully.", process.argv[2], process.argv[3])