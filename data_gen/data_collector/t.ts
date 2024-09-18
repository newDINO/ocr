/// <reference types="@types/wicg-file-system-access" />

main();

function main() {
    // Canvas
    let canvas = document.getElementById('canvas') as HTMLCanvasElement;
    let scale = 2;
    canvas.width = 256;
    canvas.height = 128;
    canvas.style.width = `${canvas.width * scale}px`;
    canvas.style.height = `${canvas.height * scale}px`;
    let ctx = canvas.getContext('2d') as CanvasRenderingContext2D;
    ctx.fillStyle = 'white';
    ctx.fillRect(0, 0, canvas.width, canvas.height);
    ctx.lineWidth = 5.0;
    ctx.lineCap = 'round';

    let lastX: number | null = null;
    let lastY: number | null = null;
    
    canvas.addEventListener('mousedown', (e) => {
        if (lastX === null) {
            lastX = e.offsetX / scale;
            lastY = e.offsetY / scale;
        }
    });
    
    canvas.addEventListener('mousemove', (e) => {
        if (lastX !== null && lastY !== null) {
            let x = e.offsetX / scale;
            let y = e.offsetY / scale;
            ctx.beginPath();
            ctx.moveTo(lastX, lastY);
            ctx.lineTo(x, y);
            lastX = x;
            lastY = y;
            ctx.stroke();
        }
    });

    canvas.addEventListener('mouseup', () => {
        lastX = null;
        lastY = null;
    });

    let clearButton = document.getElementById('clearButton') as HTMLButtonElement;
    clearButton.addEventListener('click', () => {
        ctx.fillStyle = 'white';
        ctx.fillRect(0, 0, canvas.width, canvas.height);
    });

    let lineWidthInput = document.getElementById('lineWidth') as HTMLInputElement;
    lineWidthInput.addEventListener('input', (e) => {
        ctx.lineWidth = parseFloat((e.target as HTMLInputElement).value);
    });

    // Data collection
    let fileCount = 0;
    let fileCountElement = document.getElementById('fileCount') as HTMLInputElement;
    fileCountElement.addEventListener('input', (e) => {
        fileCount = parseInt((e.target as HTMLInputElement).value);
    });

    let labels = '';

    let labelInput = document.getElementById('label') as HTMLInputElement;

    let dir: FileSystemDirectoryHandle | null = null;
    let dirPath = document.getElementById('dirPath') as HTMLSpanElement;

    let dirPicker = document.getElementById('dirPicker') as HTMLButtonElement;
    dirPicker.addEventListener('click', () => {
        showDirectoryPicker().then((directory) => {
            dir = directory;
            dirPath.textContent = dir.name;
            dir.getFileHandle('text.txt').then((file) => {
                return file.getFile();
            }).then((file) => {
                return file.text();
            }).then((text) => {
                labels = text;
                labelsTextarea.value = labels;
                fileCount = labels.split('\n').length - 1;
                fileCountElement.value = `${fileCount}`;
            });
        });
    });

    let labelsTextarea = document.getElementById('labels') as HTMLTextAreaElement;
    labelsTextarea.addEventListener('input', (e) => {
        labels = (e.target as HTMLTextAreaElement).value;
    });

    let saveButton = document.getElementById('saveButton') as HTMLButtonElement;
    saveButton.addEventListener('click', () => {
        if (dir === null) {
            return;
        }

        let label = labelInput.value;
        labels += `${label}\n`;
        labelsTextarea.value = labels;

        canvas.toBlob((blob) => {
            if (blob == null || dir == null) {
                return;
            }
            dir.getFileHandle(`${fileCount}.png`, { create: true }).then((file) => {
                file.createWritable().then((writer) => {
                    writer.write(blob);
                    writer.close();
                });
            });
        });
        fileCount++;
        fileCountElement.value = `${fileCount}`;
    });

    let saveLabelButton = document.getElementById('saveLabelButton') as HTMLButtonElement;
    saveLabelButton.addEventListener('click', () => {
        if (dir == null) {
            return;
        }
        
        let blob = new Blob([labels], { type: 'text/plain' });
        dir.getFileHandle('text.txt', { create: true }).then((file) => {
            file.createWritable().then((writer) => {
                writer.write(blob);
                writer.close();
            });
        });
    });
}