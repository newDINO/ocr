main();

async function main() {
    let session = await ort.InferenceSession.create('model.onnx');

    let canvas = document.getElementById('canvas');
    let clearButton = document.getElementById('clearButton');
    let imageInput = document.getElementById('imageInput');
    let processButton = document.getElementById('processButton');
    let outputsDiv = document.getElementById('outputsDiv');

    let scale = 2;
    canvas.width = 256;
    canvas.height = 128;
    canvas.style.width = canvas.width * scale + 'px'
    canvas.style.height = canvas.height * scale + 'px'
    let ctx = canvas.getContext('2d');

    // drawing
    ctx.lineWidth = 2.0;
    let last_x = null;
    let last_y = null;
    canvas.addEventListener('pointerdown', (e) => {
        last_x = e.offsetX / scale;
        last_y = e.offsetY / scale;
    });
    canvas.addEventListener('pointerup', () => {
        last_x = null;
        last_y = null;
    });
    canvas.addEventListener('pointermove', (e) => {
        if(last_x == null) return;
        let x = e.offsetX / scale;
        let y = e.offsetY / scale;
        ctx.beginPath();
        ctx.moveTo(last_x, last_y);
        ctx.lineTo(x, y);
        ctx.stroke();
        last_x = x;
        last_y = y;
    });
    clearButton.addEventListener('click', () => ctx.clearRect(0, 0, canvas.width, canvas.height));
    
    // putting image
    imageInput.addEventListener('change', () => {
        let file = imageInput.files.item(0);
        createImageBitmap(file)
        .then((image) => {
            let aspect = image.width / image.height;
            if(aspect > canvas.width / canvas.height) {
                let dw = canvas.width;
                let dh = dw / aspect;
                let dx = 0;
                let dy = 0.5 * (canvas.height - dh);
                ctx.drawImage(image, dx, dy, dw, dh);
            } else {
                let dh = canvas.height;
                let dw = dh * aspect;
                let dy = 0;
                let dx = 0.5 * (canvas.width - dw);
                ctx.drawImage(image, dx, dy, dw, dh);
            }
        });
    });

    // recognition
    processButton.addEventListener('click', () => {
        outputsDiv.innerHTML = '';
        generate(session, canvas, (c) => outputsDiv.innerHTML += c, 16, 0);
    });
}

async function generate(session, image, outStream, maxLen, eosId) {
    let idx = new BigInt64Array(1);
    idx[0] = BigInt(0);
    let imageTensor = ort.Tensor.fromImage(
        await createImageBitmap(image),
        {
            dataType: "float32",
            tensorFormat: "RGB",
            tensorLayout: "NCHW",
        }
    );
    console.log("generation started");

    for(let i = 0; i < maxLen; i += 1) {
        let idxTensor = new ort.Tensor(idx);
        let output = await session.run({
            "idx": idxTensor,
            "image": imageTensor,
        });
    
        let outputId = output["output"].getData();
        if(outputId[0] == eosId) {
            break;
        }
        let outputText = decode(outputId);
        idx = catBigIntArray(idx, outputId);
        console.log(outputText);
        outStream(outputText);
    }
    console.log("generation ended");
}

function decode(idx) {
    let chars = [];
    for(id of idx) {
        chars.push(id + 32);
    }
    return String.fromCharCode(chars);
}

function catBigIntArray(a1, a2) {
    let result = new BigInt64Array(a1,length + a1.length);
    result.set(a1);
    result.set(a2, a1.length);
    return result;
}