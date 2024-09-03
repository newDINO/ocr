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
    ctx.fillStyle = 'white';
    ctx.fillRect(0, 0, canvas.width, canvas.height);

    // drawing
    ctx.lineWidth = 5.0;
    ctx.lineCap = "round";
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
    clearButton.addEventListener('click', () => {
        ctx.fillStyle = 'white';
        ctx.fillRect(0, 0, canvas.width, canvas.height);
    });
    
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
    let tokenizer = new Tokenizer();
    processButton.addEventListener('click', () => {
        outputsDiv.innerHTML = '';
        image = ctx.getImageData(0, 0, canvas.width, canvas.height);
        generate(
            session,
            image,
            (c) => outputsDiv.innerHTML += c,
            48,
            tokenizer.special_token_ids['<begin>'],
            tokenizer.special_token_ids['<eos>'],
            tokenizer,
        );

        // test(session, canvas);
    });
}


async function generate(session, image, outStream, maxLen, beginId, eosId, tokenizer) {
    let idx = new BigInt64Array(1);
    idx[0] = BigInt(beginId);
    let imageTensor = await ort.Tensor.fromImage(
        image,
        {
            dataType: "float32",
            tensorFormat: "RGB",
            tensorLayout: "NCHW",
        }
    );

    for(let i = 0; i < maxLen; i += 1) {
        let idxTensor = new ort.Tensor(idx, [1, i + 1]);
        let output = await session.run(
            {
                "idx": idxTensor,
                "image": imageTensor,
            },
            {
                "logSeverityLevel": 4,
                "logVerbosityLevel": 10,
            }
        );
    
        let outputArray = await output["output"].getData();
        let outputId = Number(outputArray[0])
        if(outputId == eosId) {
            break;
        }
        let outputText = tokenizer.decode_single(outputId);
        idx = catBigIntArray(idx, outputArray);
        console.log(outputText, outputId);
        outStream(outputText);
    }
    console.log("generation ended");
}


function catBigIntArray(a1, a2) {
    let result = new BigInt64Array(a1.length + a2.length);
    result.set(a1);
    result.set(a2, a1.length);
    return result;
}

class Tokenizer {
    constructor() {
        this.ascii_char_len = 126 - 32 + 1;
        this.special_token_ids = {
            "<begin>": this.ascii_char_len,
            "<eos>": this.ascii_char_len + 1,
        };
        this.special_id_tokens = {}
        for(let token in this.special_token_ids) {
            let id = this.special_token_ids[token];
            this.special_id_tokens[id] = token;
        }
        this.vocab_size = this.ascii_char_len + Object.keys(this.special_id_tokens).length;
    }
    decode(idx) {
        let result = ''
        for(let index of idx) {
            if(index >= this.ascii_char_len) {
                result += this.special_id_tokens[index];
            } else {
                result += String.fromCharCode(index + 32);
            }
        }
        return result
    }
    decode_single(index) {
        if(index >= this.ascii_char_len) {
            return this.special_id_tokens[index];
        } else {
            return String.fromCharCode(index + 32);
        }
    }
}