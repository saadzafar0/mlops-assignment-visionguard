const input = document.getElementById("image");
const preview = document.getElementById("preview");
let previewURL;
input?.addEventListener("change", () => {
  const file = input.files[0];
  if (previewURL) URL.revokeObjectURL(previewURL);
  document.getElementById("selection").textContent = file ? `${file.name} · ${(file.size / 1024).toFixed(0)} KB` : "No image selected";
  preview.hidden = !file;
  if (file) { previewURL = URL.createObjectURL(file); preview.src = previewURL; }
});
document.getElementById("upload-form")?.addEventListener("submit", () => {
  const button = document.getElementById("check-button");
  button.disabled = true;
  button.textContent = "Checking image…";
  document.getElementById("progress-message").textContent = "Please keep this page open while your image is checked.";
});
