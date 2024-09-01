let fs = require('fs')

require('mathjax').init({
    loader: {load: ['input/tex', 'output/svg']}
}).then((MathJax) => {
    let latex_text = fs.readFileSync('data/latex/texts.txt').toString();
    let lateses = latex_text.split('\n').slice(0, -1);

    let svgs = []
    for(latex of lateses) {
        let svg = MathJax.tex2svg(latex, {display: true});
        svgs.push(MathJax.startup.adaptor.innerHTML(svg));
    }

    let out_file = fs.openSync('data/latex/svg.txt', 'w');
    fs.writeSync(out_file, svgs.join('\n'));
}).catch((err) => console.log(err.message));