main();
function main() {
  let canvas = document.getElementById("canvas");
  let scale = 2;
  canvas.width = 256;
  canvas.height = 128;
  canvas.style.width = `${canvas.width * scale}px`;
  canvas.style.height = `${canvas.height * scale}px`;
  let ctx = canvas.getContext("2d");
  ctx.fillStyle = "white";
  ctx.fillRect(0, 0, canvas.width, canvas.height);
  ctx.lineWidth = 5;
  ctx.lineCap = "round";
  let lastX = null;
  let lastY = null;
  canvas.addEventListener("mousedown", (e) => {
    if (lastX === null) {
      lastX = e.offsetX / scale;
      lastY = e.offsetY / scale;
    }
  });
  canvas.addEventListener("mousemove", (e) => {
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
  canvas.addEventListener("mouseup", () => {
    lastX = null;
    lastY = null;
  });
  let clearButton = document.getElementById("clearButton");
  clearButton.addEventListener("click", () => {
    ctx.fillStyle = "white";
    ctx.fillRect(0, 0, canvas.width, canvas.height);
  });
  let lineWidthInput = document.getElementById("lineWidth");
  lineWidthInput.addEventListener("input", (e) => {
    ctx.lineWidth = parseFloat(e.target.value);
  });
  let fileCount = 0;
  let fileCountElement = document.getElementById("fileCount");
  fileCountElement.addEventListener("input", (e) => {
    fileCount = parseInt(e.target.value);
  });
  let labels = "";
  let labelInput = document.getElementById("label");
  let dir = null;
  let dirPath = document.getElementById("dirPath");
  let dirPicker = document.getElementById("dirPicker");
  dirPicker.addEventListener("click", () => {
    showDirectoryPicker().then((directory) => {
      dir = directory;
      dirPath.textContent = dir.name;
      dir.getFileHandle("text.txt").then((file) => {
        return file.getFile();
      }).then((file) => {
        return file.text();
      }).then((text) => {
        labels = text;
        labelsTextarea.value = labels;
        fileCount = labels.split("\n").length - 1;
        fileCountElement.value = `${fileCount}`;
      });
    });
  });
  let labelsTextarea = document.getElementById("labels");
  labelsTextarea.addEventListener("input", (e) => {
    labels = e.target.value;
  });
  let saveButton = document.getElementById("saveButton");
  saveButton.addEventListener("click", () => {
    if (dir === null) {
      return;
    }
    let label = labelInput.value;
    labels += `${label}
`;
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
  let saveLabelButton = document.getElementById("saveLabelButton");
  saveLabelButton.addEventListener("click", () => {
    if (dir == null) {
      return;
    }
    let blob = new Blob([labels], { type: "text/plain" });
    dir.getFileHandle("text.txt", { create: true }).then((file) => {
      file.createWritable().then((writer) => {
        writer.write(blob);
        writer.close();
      });
    });
  });
}
