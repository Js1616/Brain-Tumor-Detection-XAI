// ===============================
// FILE INPUT + PREVIEW
// ===============================

const fileInput = document.getElementById("fileInput");
const uploadBox = document.querySelector(".upload-box");
const label = document.querySelector(".upload-box label");

// Create preview image dynamically
let previewImg = document.createElement("img");
previewImg.style.width = "200px";
previewImg.style.marginTop = "15px";
previewImg.style.borderRadius = "8px";

// When file is selected
fileInput.addEventListener("change", function () {
    const file = this.files[0];

    if (file) {
        const reader = new FileReader();

        reader.onload = function (e) {
            previewImg.src = e.target.result;

            // Remove old preview if exists
            if (!uploadBox.contains(previewImg)) {
                uploadBox.appendChild(previewImg);
            }

            label.textContent = file.name;
        };

        reader.readAsDataURL(file);
    }
});


// ===============================
// DRAG & DROP FUNCTIONALITY
// ===============================

uploadBox.addEventListener("dragover", function (e) {
    e.preventDefault();
    uploadBox.style.borderColor = "#2e7df6";
});

uploadBox.addEventListener("dragleave", function () {
    uploadBox.style.borderColor = "rgba(255,255,255,0.4)";
});

uploadBox.addEventListener("drop", function (e) {
    e.preventDefault();

    const file = e.dataTransfer.files[0];
    fileInput.files = e.dataTransfer.files;

    if (file) {
        const reader = new FileReader();

        reader.onload = function (e) {
            previewImg.src = e.target.result;

            if (!uploadBox.contains(previewImg)) {
                uploadBox.appendChild(previewImg);
            }

            label.textContent = file.name;
        };

        reader.readAsDataURL(file);
    }
});


// ===============================
// FORM VALIDATION
// ===============================

const form = document.querySelector("form");

form.addEventListener("submit", function (e) {
    if (!fileInput.files.length) {
        e.preventDefault();
        alert("Please upload an MRI image first!");
    }
});


// ===============================
// LOADING EFFECT (OPTIONAL)
// ===============================

form.addEventListener("submit", function () {

    const button = form.querySelector("button");

    button.textContent = "Analyzing...";
    button.disabled = true;

});