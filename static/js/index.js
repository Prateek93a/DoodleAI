const saveBtn = document.getElementById("save");
const clearBtn = document.getElementById("clear");
const resultp = document.getElementById("result");
let isProcessing;

saveBtn.addEventListener('click', (e) => {
    e.preventDefault();
    saveBtn.innerText = 'Wait';
    resultp.innerText = '';
    isProcessing = true;
    img = get();
    img.resize(28, 28);
    img.loadPixels();
    input = [];
    for (let i = 0; i < img.pixels.length; i += 4) {
        input.push(img.pixels[i]);
    }
    let url = 'http://localhost:5000/predict';
    let postData = { image: input };
    fetch(url, {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json'
        },
        body: JSON.stringify(postData)
    })
        .then(response => response.json())
        .then(data => {
            console.log(data);
            saveBtn.innerText = 'Predict';
            isProcessing = false;
            resultp.innerText = `I believe it is ${data.number}`;
        });
});

clearBtn.addEventListener('click', (e) => {
    resultp.innerText = '';
    background(0);
});

function setup() {
    isProcessing = false;
    resultp.innerText = '';
    createCanvas(280, 280);
    background(0);
}

function draw() {
    strokeWeight(24);
    stroke(255);
    if (!isProcessing) {
        if (mouseIsPressed) {
            line(pmouseX, pmouseY, mouseX, mouseY);
        }
    }
}